. ./path.sh

python ./wekws/bin/static_quantize.py \
    --config ./conf/tcn.yaml \
    --test_data ./data/test/data.list \
    --checkpoint /data/user/yswang/task/wekws/exp/hi_xiaowen_tcn/avg_30.pt \
    --script_model avg_30_output.pt \
    --num_workers 2