# #dowload
# download_dir=/data/user/yswang/task/wekws/data/mobvoihotwords/
# dir=/data/user/yswang/task/wekws/exp/hi_xiaowen_ds_tcn/
# mkdir ${dir}
# #./run.sh --stage 4 --stop_stage 4 --download_dir ${download_dir} --dir ${dir} >> ${dir}/log.txt 2>&1
# ./run.sh --stage 4 --stop_stage 4 --download_dir ${download_dir} --dir ${dir}

# tcn
#dowload
download_dir=/data/user/yswang/task/wekws/data/mobvoihotwords/
dir=/data/user/yswang/task/wekws/exp/hi_xiaowen_tcn/
mkdir ${dir}
#./run.sh --stage 4 --stop_stage 4 --download_dir ${download_dir} --dir ${dir} >> ${dir}/log.txt 2>&1
./run_tcn.sh --stage 2 --stop_stage 4 --download_dir ${download_dir} --dir ${dir} >> ${dir}/log.txt 2>&1