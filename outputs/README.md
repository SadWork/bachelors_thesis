# Вычислительный эксперимент

| Результат статьи | Код и сохранённые измерения |
|---|---|
| Полные модели на MNIST и CIFAR-10, табл. 1 | [MNIST](mnist_depth_mean_compare/summary.json), [Dense CIFAR-10](cifar10_dense_2000/summary.csv), [маршрутизация CIFAR-10](cifar10_deep_shared_scalar_2000/summary.csv) |
| Ранг 8 и положение BiReLU, табл. 2 | [Factor Dense на CIFAR-10](cifar10_compact_routing_800/summary.csv), [маршрутизация на CIFAR-10](cifar10_logit_placement_1200/summary.csv), [Fashion-MNIST](compact_routing_paper_check/summary.json), [CIFAR-100](compact_routing_article/independent/eval_frozen/results.json), [BiReLU после суммы на CIFAR-100](cifar100_post/) |
| Точность после сжатия, табл. 3–4 | [Сводные данные](compact_routing_article/evidence.csv), [CIFAR-10](compact_routing_article/support/test_results.json), [Fashion-MNIST](compact_routing_article/support/fashion_transfer_results.json), [CIFAR-100](compact_routing_article/independent/eval_frozen/results.json), [абляция статистик хвоста](compact_routing_article/theorem/fashion_moments.json) |
| Задержка, память и эквивалентность, табл. 5 | [Реализация C++ и сборка](compact_routing_article/systems/), [замеры](compact_routing_article/systems/timing_final.json), [численная проверка](compact_routing_article/systems/fidelity_final.json) |
| Контроль с INT8, табл. 6 | [CIFAR-10](compact_routing_article/matched_dense/eval_cifar10/results.json), [CIFAR-100](compact_routing_article/matched_dense/eval_cifar100/results.json) |

## Данные и среда

Код использует Python 3.12, NumPy, PyTorch и Matplotlib. Реализация замера задержки использует macOS на ARM и Accelerate. Путь CUDA для обучения использует Triton.

| Набор | Источник | Путь для скриптов оценки |
|---|---|---|
| CIFAR-10 | [Архив Python и описание](https://www.cs.toronto.edu/~kriz/cifar.html) | `work/cifar10_shared_scalar/data/cifar-10-python.tar.gz` |
| CIFAR-100 | [Архив Python и описание](https://www.cs.toronto.edu/~kriz/cifar.html) | `work/compact_routing_article/cifar100_data/cifar-100-python.tar.gz` |
| Fashion-MNIST | [Оригинальные файлы IDX](https://github.com/zalandoresearch/fashion-mnist#download-the-dataset) | `work/compact_routing_paper_check/fashion/` |
| MNIST | [Архивы IDX](https://storage.googleapis.com/cvdf-datasets/mnist/) | Каталог с четырьмя файлами IDX, передаваемый как `--data` |

Манифесты экспериментов фиксируют контрольные суммы файлов и разбиений. Разбиения, гиперпараметры, истории обучения и контрольные точки для проверки компилятора лежат вместе со скриптами соответствующих серий. Основная оценка использует seed 0, 1 и 2. В табл. 1 приведены сравнения с одним seed.

## Проверка компилятора и построение графика

Команды запускаются из корня репозитория.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r outputs/requirements.txt

python outputs/compact_routing_article/summarize.py
bash outputs/compact_routing_article/systems/build.sh
mkdir -p runs

VECLIB_MAXIMUM_THREADS=1 python outputs/compact_routing_article/systems/benchmark.py \
  --fidelity-only --split test \
  --refit outputs/compact_routing_article/support/focus32_topk_refit_ridge_seed0.pt \
  --out runs/fidelity.json

VECLIB_MAXIMUM_THREADS=1 python outputs/compact_routing_article/systems/benchmark.py \
  --timing-only \
  --refit outputs/compact_routing_article/support/focus32_topk_refit_ridge_seed0.pt \
  --out runs/timing.json
```

`summarize.py` строит [график зависимости точности и задержки от размера](compact_routing_article/evidence.png) и сводную таблицу из сохранённых результатов. Замеры и сверка предсказаний запускаются отдельно. На другой машине абсолютное время будет отличаться.

## Обучение

Основной модуль [train.py](compact_routing_paper_check/train.py) запускает четыре архитектуры — полный и факторизованный Dense, маршрутизацию с BiReLU до суммы и после неё. Пример для трёх seed на Fashion-MNIST.

```sh
python outputs/compact_routing_paper_check/train.py \
  --dataset fashion --data work/compact_routing_paper_check/fashion \
  --out runs/fashion --device mps --epochs 500 --ranks 8 --seeds 0 1 2 \
  --methods full_dense factor_dense logit_pre logit_post
```

Для CIFAR-100 используется [отдельный модуль с тем же интерфейсом](compact_routing_article/independent/train.py), где бюджет составляет 400 эпох и аргумент `--dataset` отсутствует. Дополнительные контроли и продолжения обучения лежат в каталогах соответствующих серий.
