import random
import sys

from hptuning import bayes_opt as hptuning


def run_experiment(model_config, hptuning_config, dataset, label):
  domain, max_trials = hptuning.get_domain(hptuning_config)
  black_box = hptuning.MLBlackBox(model_config, domain, dataset)

  report_file = 'log/randomsearch_evaluations_{}.log'.format(label)
  var_names = []
  for sub_domain in domain:
    var_names.append(sub_domain['name'])
  var_names = '\t'.join(var_names)
  with open(report_file, 'w') as f:
    f.write('iteration\teval_loss\t{}\n'.format(var_names))

  for i in range(int(max_trials)):
    hparams = [[]]
    for sub_domain in domain:
      hparams[0].append(random.choice(sub_domain['domain']))
    eval_loss = black_box.f(hparams)
    with open(report_file, 'a') as f:
      hparam_str = '\t'.join([str(hp) for hp in hparams[0]])
      f.write('{}\t{}\t{}\n'.format(i + 1, eval_loss, hparam_str))


def main():
  args = hptuning._parse_arguments(sys.argv[1:])
  hptuning._set_logging(args.log_level.upper())
  run_experiment(args.model_config, args.hptuning_config,
                 args.dataset, args.label)


if __name__ == '__main__':
  main()
