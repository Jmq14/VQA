DATA_DIR=/media/bulk/mldata/mengqinj
python -m student_code.main \
	--model coattention \
	--train_image_dir $DATA_DIR/features/train2014/ \
	--train_question_path $DATA_DIR/OpenEnded_mscoco_train2014_questions.json \
       	--train_annotation_path $DATA_DIR/mscoco_train2014_annotations.json \
	--test_image_dir $DATA_DIR/features/val2014/ \
	--test_question_path $DATA_DIR/OpenEnded_mscoco_val2014_questions.json \
	--test_annotation_path $DATA_DIR/mscoco_val2014_annotations.json \
	--batch_size 300 \
	--num_epochs 10 \
	--num_data_loader_workers 0

