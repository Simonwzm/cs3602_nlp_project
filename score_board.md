## BiLSTM

```bash
Evaluation:     Epoch: 99       Time: 0.5489    Dev acc: 69.16  Dev fscore(p/r/f): (79.86/71.95/75.70)
FINAL BEST RESULT:      Epoch: 4        Dev loss: 0.6518        Dev acc: 71.6201       Dev fscore(p/r/f): (81.4433/74.1397/77.621       Dev fscore(p1       Dev fscore(p/r/f): (81.4433/74.1397/77.6201)
```

## BertOnly 
dev acc = 78.44
dev f1 = 82.02


## Bert + GRU
dev acc = 78.77
dev f1 = 82.47
epoch = 64

## Bert + LSTM
dev acc = 79.22
dev f1 = 82.17
epoch = 95

## Bert + CRF
dev acc = 78.66
dev f1 = 82.06
epoch = 85


## Bert + LSTM + augmentation
dev acc = 77.88
dev f1 = 81.86
epoch = 38

reason: many unrelated output

## Bert + augmentation
dev acc = 78.32
dev f1 = 81.76
epoch = 52

## baseline + augmentation
dev acc = 77.09
dev f1 = 80.61
epoch = 99

## baseline + old augmentation
dev acc = TBD

## baseline + import dictionary
dev acc = 73.74
dev f1 = 76.68
epoch = 100







## Bert+Multi-task
dev acc = 78.2
dev f1 = 80.95 
epoch = 87

## Bert+Multi-task+augmentation
dev acc = 77.32 TBD
dev f1 = 80.91
epoch = 73





## Flat Lattice
fscore=0.7660239708181344, P=0.765625, R=0.7664233576642335


