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
  { test $ALG && test $PARAM && test $DSIZE; } || { echo "alg or param or dsize is not set"; return; }
  { test $bs && test $lr; } || { echo "bs or lr is not set"; return; }
  # continue here only if they are set
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
for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$m-$i.csv -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -y $YLIST --reference $DATADIR/$ALG-$PARAM-$DSIZE-p.txt -o $SCRDIR/$m-$i.txt
done done

# calculate and evaluation (using 6 thread)
ALG=rnn;PARAM=10,4r3,1;bs=3000;lr=0.1;
for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date); set_dir
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -r $RESDIR/$m-$i.csv -y $YLIST -o gd:$lr -s $bs --term_iter $ITER --term_time $TIME --arch_iter 1000 --arch_time 0.5 --log_iter 200 --v=1 > $LOGDIR/$m-$i;
echo "  postprocess";
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$m-$i.csv -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -y $YLIST --reference $DATADIR/$ALG-$PARAM-$DSIZE-p.txt -o $SCRDIR/$m-$i.txt -w 6;
done done

# priority
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -r $RESDIR/$m-$i-p$k.csv -y $YLIST -o psgd:$lr:$k -s $bs -b --term_iter $ITER --term_time $TIME --arch_iter 10 --arch_time 20 --log_iter 100 --v=1 > $LOGDIR/$m-$i-p$k;


# batch of priority and benchmark
ITER=10k; TIME=600;
bs0=60000; for k0 in 0.01 0.05 0.1 0.15 0.2; do for i in 4; do 
# benchmark - change bs
bs=$(echo $bs0*$k0/1 | bc); echo bench-$bs-$m-$i - $(date); set_dir;
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d /var/tzhou/data/mnist/mnist_train.csv -r $RESDIR/$m-$i.csv -y $YLIST -o gd:$lr -s $bs -b --term_iter 10k --term_time 600 --arch_iter 10 --arch_time 20 --log_iter 100 --v=2 > $LOGDIR/$m-$i;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$m-$i.csv -d /var/tzhou/data/mnist/mnist_train.csv -y $YLIST -b -o $SCRDIR/$m-$i.txt -w 6;
# priority - change k
bs=$bs0; k=$(echo $k0/$i | bc -l | sed 's/0\+$//' | sed 's/^\./0./'); echo bench-$bs-$m-$i-$k - $(date); set_dir;
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d /var/tzhou/data/mnist/mnist_train.csv -r $RESDIR/$m-$i-p$k.csv -y $YLIST -o psgd:$lr:$k -s $bs -b --term_iter 10k --term_time 600 --arch_iter 10 --arch_time 20 --log_iter 100 --v=2 > $LOGDIR/$m-$i-p$k;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$m-$i-p$k.csv -d /var/tzhou/data/mnist/mnist_train.csv -y $YLIST -b -o $SCRDIR/$m-$i-p$k.txt -w 6;
done done

