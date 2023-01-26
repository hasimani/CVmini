mkdir outputs
for sample in test/*
do
    if [[ $sample == *.mp4 ]]
    then
        python label_video.py --video $sample

        mv test/*.avi outputs/

    fi
done