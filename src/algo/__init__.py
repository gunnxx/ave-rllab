from src.algo.grbal import GrBAL
from src.algo.quick_adaptation.famle import FAMLE
from src.algo.quick_adaptation.grbal_diff_mse import GrBALDiffMSE

REGISTERED_ALGO = {
  "famle": FAMLE,
  "grbal": GrBAL,
  "grbal_diff_mse": GrBALDiffMSE
}
