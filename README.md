## Style and API checklist
- Type-hint for every variables in constructor.
- `REQUIRED_CONFIG_KEYS: Dict[str, Any]` is required for every non-abstract classes where the values are the exact copy of the constructor parameters. It is needed to populate the `Config` for missing values with default values.
- `REQUIRED_*: Dict[str, str]` is required for every keys in `REQUIRED_CONFIG_KEYS` whose values will be non-JSON datatype i.e. `algo`, `model`, and `buffer`. The key of `REQUIRED_*` should be registered in `REGISTERED_*` () and the corresponding value is the key in `REQUIRED_CONFIG_KEYS` that will become the parameters to construct the non-JSON datatype.
- Import should start from `src` e.g. `from src.buffer.buffer import Buffer`.
- Each non-abstract class have non-static `validate_params` method to be called by `@classmethod instantiate_*` on the base class to validate parameters passed.
- After creating non-abstract class, dont forget to register on the module's `__init__.py`.

## (Algorithm) to dos:
[ ] `GrBAL` (original, nested, and first-order)
[ ] `FAMLE`
[ ] `MAML + On-Policy RL`
[ ] `ReBAL`

## (General) to dos:
[ ] Add `mpc_controller` module. It will not be a `class`, but `Callable`. Style follows other modules e.g. `buffer` module. Parameters should be all the same since all MPC share common structures i.e. model, time-horizon, reward, etc.
[ ] Complete `config.py`
[ ] Complete `run.py`.
[ ] Add `device` option.
[ ] Create `Logger` on utils.
[ ] Create `Renderer` on utils.
[ ] (Optional) Add Data-Parallel capabilities.