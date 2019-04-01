MODE="bsp tap fsp aap"

DATADIR=/tmp/tzhou/data
DATADIR=/var/tzhou/data
RESBASEDIR=~/Code/FSB/result/
SCRBASEDIR=~/Code/FSB/score/
LOGBASEDIR=~/Code/FSB/log/
MPICMD="mpirun -n $((i+1))"

DATASET=mnist
RESBASEDIR=~/efs/result/$DATASET
SCRBASEDIR=~/efs/score/$DATASET
LOGBASEDIR=~/efs/log/$DATASET
MPICMD="mpirun -mca btl ^openib -n $((i+1))"

ALG=lr
PARAM=10

#ALG=mlp
#PARAM=10,15,1

#ALG=cnn
#PARAM=100,20c5p2,1c5p2,1

#ALG=rnn
#PARAM=100,10rs5,1

YLIST=10

DSIZE=100k
#DF=$ALG-$PARAM-$DSIZE-d.csv

bs=100 # 1000 10000
lr=0.1 # 0.01
ITER=100k
TIME=60

function set_dir(){
  export RESDIR=$RESBASEDIR/$ALG/$PARAM-$DSIZE/$bs-$lr
  export SCRDIR=$SCRBASEDIR/$ALG/$PARAM-$DSIZE/$bs-$lr
  export LOGDIR=$LOGBASEDIR/$ALG/$PARAM-$DSIZE/$bs-$lr
  mkdir -p $RESDIR
  mkdir -p $SCRDIR
  mkdir -p $LOGDIR
}

set_dir

cd ~/Code/FSB/FAB/build

for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -r $RESDIR/$m-$i.csv -y $YLIST -o gd:$lr -s $bs --term_iter $ITER --term_time $TIME --arch_iter 1000 --arch_time 0.5 --log_iter 200 --v=1 > $LOGDIR/$m-$i;
#mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -r $RESDIR/$m-$i.csv -y $YLIST -o em:$lr -s $bs --term_iter $ITER --term_time $TIME --arch_iter 1000 --arch_time 0.5 --log_iter 200 --v=1 > $LOGDIR/$m-$i;
done done

for PARAM in 10,1; do echo $PARAM;
for bs in 10000 1000 100; do for lr in 0.1 0.01; do echo $bs - $lr; set_dir;
for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -r $RESDIR/$m-$i.csv -y $YLIST -o gd:$lr -s $bs --term_iter $ITER --term_time $TIME --arch_iter 1000 --arch_time 0.5 --log_iter 200 --v=1 > $LOGDIR/$m-$i;
done done
done done
done

# evaluate
PATH=$PATH:/home/tzhou/Code/FSB/FAB/build/Release/src/main

for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
postprocess -a $ALG -p $PARAM -r $RESDIR/$m-$i.csv -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -y $YLIST --reference $DATADIR/$ALG-$PARAM-$DSIZE-p.txt -o $SCRDIR/$m-$i.txt
done done

# calculate and evaluation (in background)
PARAM=10,4r3,1;bs=3000;lr=0.1;
for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date); set_dir
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -r $RESDIR/$m-$i.csv -y $YLIST -o gd:$lr -s $bs --term_iter $ITER --term_time $TIME --arch_iter 1000 --arch_time 0.5 --log_iter 200 --v=1 > $LOGDIR/$m-$i;
echo "  postprocess";
postprocess -a $ALG -p $PARAM -r $RESDIR/$m-$i.csv -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -y $YLIST --reference $DATADIR/$ALG-$PARAM-$DSIZE-p.txt -o $SCRDIR/$m-$i.txt &
done done

