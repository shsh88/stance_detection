package ude.master.thesis.stance_detection.main;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;

import ude.master.thesis.stance_detection.ml.MainClassifier;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.classifiers.functions.LibSVM;

public class StanceWord2VecClassifier {

	public static final boolean USE_TRAINING_SET = true;
	public static final boolean USE_TEST_SET = true;

	public static void main(String[] args) throws IOException {
		long start = System.currentTimeMillis();

		StanceDetectionDataReader datasetReader = new StanceDetectionDataReader(USE_TRAINING_SET, USE_TEST_SET);

		Map<Integer, String> trainingIdBodyMap = datasetReader.getTrainIdBodyMap();
		List<List<String>> trainingStances = datasetReader.getTrainStances();

		Map<Integer, String> testIdBodyMap = datasetReader.getTestIdBodyMap();
		List<List<String>> testStances = datasetReader.getTestStances();

		// SMO smo = new SMO();

		MainClassifier classifier = new MainClassifier(trainingIdBodyMap, trainingStances, testIdBodyMap, testStances,
				new LibSVM());
		/*
		 * DataSource trainDataSource = new DataSource(
		 * "resources/arff_data/BoW_features_Chi_thre0_500008-11_10-36.arff");
		 * Instances trainData = trainDataSource.getDataSet();
		 * trainData.setClassIndex(trainData.numAttributes() - 1);
		 * classifier.setTrainingInstances(trainData);
		 * 
		 * DataSource testDataSource = new DataSource(
		 * "resources/arff_data/BoW_features_Chi_thre0_500008-11_10-36_test.arff"
		 * ); Instances testData = testDataSource.getDataSet();
		 * testData.setClassIndex(testData.numAttributes() - 1);
		 * classifier.setTestInstances(testData);
		 */
		classifier.setUseAttributeSelectionFilter(false);
		//classifier.setBOW_useLemmatization(true);
		//classifier.setEvaluate(false);
		classifier.setUseParagraphsEmbeddings(true);
		classifier.evaluate();
		classifier.saveInstancesToArff("Doc2Vec_features_" + getCurrentTimeStamp());

		// classifier.train();

		System.out.println(System.currentTimeMillis() - start);
	}

	public static String getCurrentTimeStamp() {
		return new SimpleDateFormat("MM-dd_HH-mm").format(new Date());
	}

}
