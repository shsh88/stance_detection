package ude.master.thesis.stance_detection.processor;

import ude.master.thesis.stance_detection.main.ClassifierTools;
import ude.master.thesis.stance_detection.util.FNCConstants;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class BoWSerialisation {

	private static Instances trainingInstances;

	private static Instances unlabtestInstances;

	public static void main(String[] args) throws Exception {

		//TODO here the training instances 38840
		trainingInstances = StanceDetectionDataReader
				.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "Ferr_doc2vec_train09-19_11-55.arff");
		unlabtestInstances = StanceDetectionDataReader
				.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "Ferr_doc2vec_unlabeled_test09-19_11-55.arff");
	
		trainingInstances.addAll(unlabtestInstances);
		
		ClassifierTools ct = new ClassifierTools(trainingInstances, unlabtestInstances, null);
		StringToWordVector bow = ct.applyBoWFilter(2000, 1, 2);
		String time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("ferr_BoW2000_allf_" + time);
		
		 weka.core.SerializationHelper.write(ProjectPaths.MODEL_PATH + "BoW_2000_train_test", bow);
	
	}

}
