import argparse

import src.algo as algo
from src.config import Config

def main(args):
  algo_class = algo.REGISTERED_ALGO[args.algo]
  
  ## Construct the `Config`.
  ## default -> experiment-default -> tiny tweak using args
  config = Config(algo_class)
  if args.json_file: config.fill_from_json(args.json_file)
  config.fill(**vars(args))

  config.save_as_json(args.exp_dir)
  config.prepare()
  
  ## Instantiate and run the algo
  algo_class.validate_params(config.data)
  algo_class(**config.data).run()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # main arguments
  parser.add_argument('--algo', type=str, help='One of REGISTERED_ALGO in src/algo/__init__.py', required=True)
  parser.add_argument('--env', type=str, help='Gym Environment name.', required=True)
  parser.add_argument('--exp_dir', type=str, help='Target directory to store experiment logs.', required=True)
  parser.add_argument('--device', type=str, help='Device used to run the algo.', default='cpu')
  parser.add_argument('--json_file', type=str, help='Path to configuration file in JSON format.', default=None)

  _, other_args = parser.parse_known_args()

  # algo-specific args
  for keyword in other_args:
    if keyword.startswith('--'):
      parser.add_argument(keyword.split('=')[0])
  
  args = parser.parse_args()

  main(args)