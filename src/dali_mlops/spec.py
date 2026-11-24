"""Static pipeline spec parser.

Standard library only. It never imports or executes the pipeline file and does not import ZenML,
so the Service Manager can vendor this single file and run it on uploaded source.

    parse_pipeline_source(text) -> {"pipelines": [...], "errors": [...], "warnings": [...]}

Each pipeline is ``{name, function, sdk, dataset_format, params, base_image, requirements,
skip_build, environment}`` and each param is ``{name, type, required, default, reserved}``.
"""

import ast

SPEC_VERSION = 1
DATASET_FORMATS = ("csv", "tsv", "parquet", "json", "jsonl", "raw")
# The run contract: parameters the Service Manager fills. Order: required first, then defaulted.
# ``dataset`` marks the ones that exist only when the pipeline reads a dataset.
RESERVED = (
    {"name": "dataset_id", "type": "str", "required": True, "default": None, "dataset": True},
    {"name": "dataspace_model_id", "type": "str", "required": True, "default": None, "dataset": False},
    {"name": "run_name", "type": "str", "required": True, "default": None, "dataset": False},
    {"name": "asset_id", "type": "str | None", "required": False, "default": None, "dataset": True},
    {"name": "experiment_name", "type": "str", "required": False, "default": "", "dataset": False},
)
RESERVED_NAMES = frozenset(item["name"] for item in RESERVED)
_UNKNOWN = object()


def reserved_params(has_dataset=True):
    return [
        {"name": item["name"], "type": item["type"], "required": item["required"], "default": item["default"],
         "reserved": True}
        for item in RESERVED if has_dataset or not item["dataset"]
    ]


class _Call:
    def __init__(self, func, args, kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs


def _last_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _value(node, env):
    """Evaluate literals, containers, simple constants and calls (as _Call). Anything else is _UNKNOWN."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, _UNKNOWN)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_value(item, env) for item in node.elts]
        return _UNKNOWN if any(item is _UNKNOWN for item in items) else items
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            return _UNKNOWN
        keys = [_value(key, env) for key in node.keys]
        values = [_value(value, env) for value in node.values]
        if any(item is _UNKNOWN for item in keys + values) or any(isinstance(key, (list, dict)) for key in keys):
            return _UNKNOWN
        return dict(zip(keys, values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _value(node.operand, env)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            return -operand if isinstance(node.op, ast.USub) else operand
        return _UNKNOWN
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _value(node.left, env), _value(node.right, env)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        if isinstance(left, list) and isinstance(right, list):
            return left + right
        return _UNKNOWN
    if isinstance(node, ast.Call):
        func = _last_name(node.func)
        if func is None:
            return _UNKNOWN
        return _Call(
            func, [_value(arg, env) for arg in node.args],
            {kw.arg: _value(kw.value, env) for kw in node.keywords if kw.arg},
        )
    return _UNKNOWN


def _plain(value):
    return None if value is _UNKNOWN or isinstance(value, _Call) else value


def _constants(tree):
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            env[node.targets[0].id] = _value(node.value, env)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            env[node.target.id] = _value(node.value, env)
    return env


def _decorator(node):
    """Return (kind, call_node_or_None) for @pipeline / @dali_pipeline decorators, else (None, None)."""
    call = node if isinstance(node, ast.Call) else None
    name = _last_name(call.func if call else node)
    return (name, call) if name in ("pipeline", "dali_pipeline") else (None, None)


def _imports_sdk(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "dali_mlops":
            return True
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "dali_mlops" for alias in node.names):
            return True
    return False


def _params(function, env, warnings, skip_first):
    args = function.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    entries = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
    if skip_first and entries:
        entries = entries[1:]
    if args.vararg or args.kwarg:
        warnings.append(f"{function.name}: *args and **kwargs are not pipeline parameters and were ignored.")
    result = []
    for arg, default in entries:
        value = None
        if default is not None:
            raw = _value(default, env)
            if raw is _UNKNOWN or isinstance(raw, _Call):
                warnings.append(f"{function.name}: the default of '{arg.arg}' is not a literal and was not read.")
            else:
                value = raw
        result.append({
            "name": arg.arg,
            "type": ast.unparse(arg.annotation) if arg.annotation is not None else None,
            "required": default is None,
            "default": value,
            "reserved": arg.arg in RESERVED_NAMES,
        })
    return result


def _order(reserved, own):
    """Required parameters first, then defaulted ones; reserved before the author's within each group."""
    items = reserved + own
    return [p for p in items if p["required"]] + [p for p in items if not p["required"]]


def _docker(settings, env):
    """Read DockerSettings(...) from settings={'docker': ...} (inline or through a module constant)."""
    docker = settings.get("docker") if isinstance(settings, dict) else None
    if isinstance(docker, _Call) and docker.func == "DockerSettings":
        return docker.kwargs
    return {}


def _pipeline(function, kind, call, env, sdk_module, warnings):
    kwargs = {}
    if call is not None:
        for keyword in call.keywords:
            if keyword.arg:
                kwargs[keyword.arg] = _value(keyword.value, env)
        if call.args and kind == "pipeline":
            kwargs.setdefault("name", _value(call.args[0], env))
    is_sdk = kind == "dali_pipeline"
    docker = {}
    dataset_format = None
    if is_sdk:
        dataset_format = kwargs.get("dataset", "csv")
        if dataset_format is _UNKNOWN or not (dataset_format is None or dataset_format in DATASET_FORMATS):
            warnings.append(f"{function.name}: dataset must be one of {', '.join(DATASET_FORMATS)} or None.")
            dataset_format = None
        base_image, requirements = kwargs.get("base_image"), kwargs.get("requirements")
        skip_build = bool(requirements) is False and base_image is not None
    else:
        docker = _docker(kwargs.get("settings"), env)
        base_image, requirements = docker.get("parent_image"), docker.get("requirements")
        skip_build = docker.get("skip_build")
        if kwargs.get("settings") is _UNKNOWN or (
            isinstance(kwargs.get("settings"), dict) and not docker and "docker" in kwargs["settings"]
        ):
            warnings.append(f"{function.name}: the docker settings are not literal and were not read.")
    environment = kwargs.get("environment")
    if environment is not None and not isinstance(environment, dict):
        warnings.append(f"{function.name}: the environment is not a literal mapping and was not read.")
    name = kwargs.get("name")
    own = _params(function, env, warnings, skip_first=is_sdk and dataset_format is not None)
    if is_sdk:
        clash = sorted(RESERVED_NAMES.intersection(p["name"] for p in own))
        if clash:
            warnings.append(f"{function.name}: parameters {clash} are reserved and set by the platform.")
            own = [p for p in own if p["name"] not in clash]
        params = _order(reserved_params(dataset_format is not None), own)
    else:
        params = own
    return {
        "name": name if isinstance(name, str) else function.name,
        "function": function.name,
        "sdk": is_sdk or sdk_module,
        "dataset_format": dataset_format,
        "params": params,
        "base_image": base_image if isinstance(base_image, str) else None,
        "requirements": requirements if isinstance(requirements, (str, list)) else None,
        "skip_build": skip_build if isinstance(skip_build, bool) else None,
        "environment": {k: v for k, v in environment.items() if isinstance(k, str) and isinstance(v, str)}
        if isinstance(environment, dict) else {},
    }


def parse_pipeline_source(text):
    result = {"pipelines": [], "errors": [], "warnings": []}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as error:
        line = getattr(error, "lineno", None)
        result["errors"].append(f"Syntax error{f' at line {line}' if line else ''}: {getattr(error, 'msg', error)}")
        return result
    env = _constants(tree)
    sdk_module = _imports_sdk(tree)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                kind, call = _decorator(decorator)
                if kind:
                    result["pipelines"].append(_pipeline(node, kind, call, env, sdk_module, result["warnings"]))
                    break
    if not result["pipelines"]:
        result["errors"].append("No pipeline found. Decorate a function with @dali_pipeline or @pipeline.")
    elif len(result["pipelines"]) > 1:
        result["warnings"].append("The file defines several pipelines.")
    return result
