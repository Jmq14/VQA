DATA_DIR=/media/bulk/mldata/mengqinj
python -m student_code.save_feature \
	--train_image_dir $DATA_DIR/train2014/ \
	--train_question_path $DATA_DIR/OpenEnded_mscoco_train2014_questions.json \
       	--train_annotation_path $DATA_DIR/mscoco_train2014_annotations.json \
	--test_image_dir $DATA_DIR/val2014/ \
	--test_question_path $DATA_DIR/OpenEnded_mscoco_val2014_questions.json \
	--test_annotation_path $DATA_DIR/mscoco_val2014_annotations.json \
	--output_path $DATA_DIR/features

