#python run_trial.py --dataset office_home pacs --train-fe --fe-trainer erm --device cuda:0 --num_epochs 100 --seed 0 > erm.log &
#python run_trial.py --dataset office_home pacs --train-fe --fe-trainer erm --device cuda:1 --num_epochs 100 --seed 0 --mixup > erm_mixup.log &
#python run_trial.py --dataset office_home pacs --train-fe --fe-trainer erm --device cuda:2 --num_epochs 100 --seed 0 --pretrained > erm_pretrained.log &
python run_trial.py --dataset pacs --train-fe --fe-trainer erm --device cuda:3 --num_epochs 100 --seed 0 --augment-data > erm_aug.log &
python run_trial.py --dataset office_home pacs --train-fe --fe-trainer erm --device cuda:0 --num_epochs 100 --seed 0 --mixup --pretrained --augment-data > erm_all.log
