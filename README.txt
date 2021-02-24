Main requirements:
pytorch, torchvision, numpy, pandas, robustness, robustbench

Train clean model:
python3 train.py --eps 0.0 --attack none --epochs 150 --data_dir ../datasets/ --model_path models/clean.pt

Train L2 AT model:
python3 train.py --eps 25.5 --attack rlat --distance l2 --epochs 150 --data_dir ../datasets/ --model_path models/l2_0.1.pt

Train RLAT model:
python3 train.py --eps 25.5 --attack rlat --distance l2 --epochs 150 --data_dir ../datasets/ --model_path models/rlat_0.1.pt

Epsilon should be specified multiplied by 255.0