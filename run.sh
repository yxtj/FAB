MODE="bsp tap fsp aap"

DATADIR=/tmp/tzhou/data
DATADIR=/var/tzhou/data
RESBASEDIR=~/Code/FSB/result/
SCRBASEDIR=~/Code/FSB/score/
LOGBASEDIR=~/Code/FSB/log/
MPICMD="mpirun -n $((i+1))"
MPICMD="mpirun --mca btl_tcp_if_include 192.168.0.0/24 -n $((i+1))"

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

#k=$(echo $k0/$i | bc -l | sed 's/0\+$//' | sed 's/^\./0./');
#lr=$(printf "%.4f" $(echo $lr0/$k | bc -l) | sed 's/0*$//')

# batch of priority and benchmark
ITER=10k; TIME=600;
bs0=60000; for k in 0.01 0.05 0.1 0.15 0.2; do for i in 4; do 
# benchmark - change bs
bs=$(echo $bs0*$k/1 | bc); echo bench-$bs-$m-$i - $(date); set_dir;
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d /var/tzhou/data/mnist/mnist_train.csv -r $RESDIR/$m-$i.csv -y $YLIST -o gd:$lr -s $bs -b --term_iter 10k --term_time 600 --arch_iter 10 --arch_time 20 --log_iter 100 --v=2 > $LOGDIR/$m-$i;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$m-$i.csv -d /var/tzhou/data/mnist/mnist_test.csv -y $YLIST -b -o $SCRDIR/$m-$i.txt -w 6;
# priority - change k
bs=$bs0; echo priority-$bs-$m-$i-$k - $(date); set_dir;
fn=$m-$i-p$k;
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d /var/tzhou/data/mnist/mnist_train.csv -r $RESDIR/$fn.csv -y $YLIST -o psgd:$lr:$k -s $bs -b --term_iter 10k --term_time 600 --arch_iter 10 --arch_time 20 --log_iter 100 --v=2 > $LOGDIR/$fn;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$fn.csv -d /var/tzhou/data/mnist/mnist_test.csv -y $YLIST -b -o $SCRDIR/$fn.txt -w 6;
done done

ARVTIME=20;
DATA_TRAIN=/var/tzhou/data/mnist/mnist_train.csv;
DATA_TEST=/var/tzhou/data/mnist/mnist_test.csv;

DO_ACCURACY="--accuracy"

bs0=60000;
for k in 0.01 0.05 0.1 0.15 0.2; do 
bs=$(echo $bs0*$k/1 | bc);
echo bench-$bs-$m-$i - $(date); set_dir; fn=$m-$i;
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATA_TRAIN -r $RESDIR/$fn.csv -y $YLIST -o gd:$lr -s $bs -b --term_iter 10k --term_time 600 --arch_iter 10 --arch_time $ARVTIME --log_iter 100 --v=2 > $LOGDIR/$fn; killall main;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$fn.csv -d $DATA_TEST -y $YLIST -b -o $SCRDIR/$fn.txt $DO_ACCURACY -w 6;
bs=$bs0; set_dir;
for r in 0 0.001 0.01 0.05; do
fn=$m-$i-pg$k-r$r; echo priority-$bs-$fn - global- $(date);
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATA_TRAIN -r $RESDIR/$fn.csv -y $YLIST -o psgd:$lr:$k:global:$r -s $bs -b --term_iter 10k --term_time 600 --arch_iter 10 --arch_time $ARVTIME --log_iter 100 --v=2 > $LOGDIR/$fn; killall main;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$fn.csv -d $DATA_TEST -y $YLIST -b -o $SCRDIR/$fn.txt $DO_ACCURACY -w 6;
fn=$m-$i-ps$k-r$r; echo priority-$bs-$fn - global- $(date);
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATA_TRAIN -r $RESDIR/$fn.csv -y $YLIST -o psgd:$lr:$k:square:$r -s $bs -b --term_iter 10k --term_time 600 --arch_iter 10 --arch_time $ARVTIME --log_iter 100 --v=2 > $LOGDIR/$fn; killall main;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$fn.csv -d $DATA_TEST -y $YLIST -b -o $SCRDIR/$fn.txt $DO_ACCURACY -w 6;
done
done

bs=60000; set_dir; 
for k in 0.01 0.05 0.1 0.15 0.2; do
dk=$(awk "BEGIN{ print 1-$k; }");
DLIST=$(echo "0.7 0.8 0.9 $dk" | tr ' ' '\n' | sort -nu)
for d in $DLIST; do
for r in 0.001 0.01 0.05; do
fn=$m-$i-p$k-r$r-d$d; echo priority-$bs-$fn - $(date);
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATA_TRAIN -r $RESDIR/$fn.csv -y $YLIST -o psgd:$lr:$k:$d:$r -s $bs -b --term_iter 10k --term_time 600 --arch_iter 10 --arch_time $ARVTIME --log_iter 100 --v=2 > $LOGDIR/$fn; killall main;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$fn.csv -d $DATA_TEST -y $YLIST -b -o $SCRDIR/$fn.txt $DO_ACCURACY -w 6;
done done
done

# loop on (k,r) pairs
for kr in 0.01,0.04 0.02,0.03 0.03,0.02 0.04,0.01; do
k=$(echo $kr | sed 's/,.*//'); r=$(echo $kr | sed 's/.*,//');
fn=$m-$i-p$k-r$r-lp-l; echo $fn - $(date);
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM -d $DATA_TRAIN -r $RESDIR/$fn.csv -y $YLIST -o psgd:$lr:$k:$r:l:p:l -s $bs -b --term_iter 10k --term_time 600 --arch_iter 10 --arch_time $ARVTIME --log_iter 100 --v=2 > $LOGDIR/$fn; killall main;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$fn.csv -d $DATA_TEST -y $YLIST -b -o $SCRDIR/$fn.txt $DO_ACCURACY -w 6;
done

# using predefined dataset: mnist
mpirun -n $((i+1)) src/main/main -m $m -a $ALG -p $PARAM --dataset mnist -d $DATA_TRAIN -r $RESDIR/$fn.csv -o psgd:$lr:$k:$r:l:p:l -s $bs -b --term_iter 10k --term_time 300 --arch_iter 10 --arch_time $ARVTIME --log_iter 100 --v=2 > $LOGDIR/$fn; killall main;
src/main/postprocess -a $ALG -p $PARAM -r $RESDIR/$fn.csv --dataset mnist -d $DATA_TEST -b -o $SCRDIR/$fn.txt $DO_ACCURACY -w 6;


