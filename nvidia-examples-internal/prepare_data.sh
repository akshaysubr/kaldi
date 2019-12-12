#!/bin/bash


rm -f  $MODEL_NAME.tar.gz

wget http://sqrl/datasets/kaldi/iv/$MODEL_NAME.tar.gz
#ln -sf /tarfiles/$MODEL_NAME.tar.gz 

tar -xzf $MODEL_NAME.tar.gz
rm -Rf $MODEL_NAME.tar.gz

mkdir -p $WORKSPACE/data/iv/
mkdir -p $WORKSPACE/datasets/iv/
mkdir -p $WORKSPACE/models/iv/

rm -Rf $WORKSPACE/data/iv/$MODEL_NAME
rm -Rf $WORKSPACE/datasets/iv/$MODEL_NAME
rm -Rf $WORKSPACE/model/iv/$MODEL_NAME

mv $MODEL_NAME/data $WORKSPACE/data/iv/$MODEL_NAME
mv $MODEL_NAME/dataset $WORKSPACE/datasets/iv/$MODEL_NAME
mv $MODEL_NAME/model $WORKSPACE/models/iv/$MODEL_NAME

rm -Rf $MODEL_NAME
# Search and replace "workspace" with the $WORKSPACE directory
# in .conf files and in wav.scp
pushd $WORKSPACE/models/iv/$MODEL_NAME/conf
for i in *.conf; do
   sed -i "s@workspace@$WORKSPACE@g" $i
done
popd
sed -i "s@workspace@$WORKSPACE@g" $WORKSPACE/datasets/iv/$MODEL_NAME/wav.scp

pushd $WORKSPACE/models/iv/$MODEL_NAME/
ln -sf graph/phones/ ./phones
popd
ln -sf ../run_benchmark.sh
