# team16_11777

Download data:
```
export fileid="1YNzlpZygXxHvZ7vIHMjyBRCykm4iWrdD"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o hard_negs.zip
rm ./cookie
```

Install requriments:
```
pip install -r requirements.txt
```

Train model: 
```
IMAGE_PATH=/path/to/images/
ANNOTATION_FILE=/path/to/annotations.json

python train_linear_head.py --image_path $IMAGE_PATH --annotation_file $ANNOTATION_FILE --use_diffusion
```
`--use_diffusion`: use Stable Diffusion to generate negative image samples.