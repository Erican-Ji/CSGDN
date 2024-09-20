# 读取目录 ./results/cotton/CSGDN/tmp_search 下的所有文件的最后一行，将其追加到 ./results/cotton/CSGDN/search.txt
for file in ./results/cotton/CSGDN/tmp_search/*
do
    echo $file >> ./results/cotton/CSGDN/search.txt
    tail -n 1 $file >> ./results/cotton/CSGDN/search.txt
done

# 找到最后一行对应为 {'acc': 0.8120300751879699, 'auc': 0.7773089545241444, 'f1': 0.7191011235955056, 'micro_f1': 0.8120300751879699, 'macro_f1': 0.7889290928711992, 'mask_ratio': 0.1, 'alpha': 0.2, 'beta': 0.01, 'res': 'Stage 4DPA: acc 0.785+0.035; auc 0.746+0.038; f1 0.668+0.060; micro_f1 0.785+0.035; macro_f1 0.755+0.042\n'} 的文件，返回其文件名
# for file in ./results/cotton/CSGDN/tmp_search/*
# do
    # tail -n 1 $file | grep "Stage 4DPA: acc 0.785+0.035; auc 0.746+0.038; f1 0.668+0.060; micro_f1 0.785+0.035; macro_f1 0.755+0.042" > /dev/null
    # if [ $? -eq 0 ]; then
        # echo $file
    # fi
# done
