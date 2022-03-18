exp=${1}
epoch=${2}
c=${3}
if [ -z "$1" ]
then
echo please specify the path to the search experiment
exit 1
fi
if [ -z "$2" ]
then
echo please specify the epoch to extract architectures from
exit 1
fi
if [ -z "$3" ]
then
echo c not specified. setting to 1.0 ...
c=1.0
fi

#sample architecture via MCTS, then print the sampled tokens to sample_arch.py
#for further training
python derive.py --exp ${exp} --epoch ${epoch} --c ${c} --print|tee ${exp}/${exp}_.log
python derive.py --exp ${exp} --epoch ${epoch} --c ${c} --print --max_sample|tee -a ${exp}/${exp}_.log
grep sample ${exp}/${exp}_.log|tee -a sample_arch.py
