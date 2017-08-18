package ude.master.thesis.stance_detection.main;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;

import ude.master.thesis.stance_detection.ml.MainClassifier;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class StanceBoWClassifierWithCombinedVec {
	public static final boolean USE_TRAINING_SET = true;
	public static final boolean USE_TEST_SET = true;

	public static void main(String[] args) throws Exception {

		long start = System.currentTimeMillis();

		StanceDetectionDataReader datasetReader = new StanceDetectionDataReader(USE_TRAINING_SET, USE_TEST_SET);

		Map<Integer, String> trainingIdBodyMap = datasetReader.getTrainIdBodyMap();
		List<List<String>> trainingStances = datasetReader.getTrainStances();

		Map<Integer, String> testIdBodyMap = datasetReader.getTestIdBodyMap();
		List<List<String>> testStances = datasetReader.getTestStances();

		// SMO smo = new SMO();

		MainClassifier classifier = new MainClassifier(trainingIdBodyMap, trainingStances, testIdBodyMap, testStances,
				new LibSVM());

				DataSource trainDataSource = new DataSource("C:/arff_data/combined_train_08-17_18-04.arff");
				Instances trainData = trainDataSource.getDataSet();
				trainData.setClassIndex(trainData.numAttributes() - 1);
				classifier.setTrainingInstances(trainData);

				DataSource testDataSource = new DataSource("C:/arff_data/combined_test_08-17_18-04.arff");
				Instances testData = testDataSource.getDataSet();
				testData.setClassIndex(testData.numAttributes() - 1);
				classifier.setTestInstances(testData);

		classifier.setUseTitle(true);
		classifier.setUseArticle(true);
		classifier.setUseAttributeSelectionFilter(true);
		classifier.setBOW_useLemmatization(true);
		
		//classifier.evaluateWithCrossValidation("BoW_features_combined_vecs");
		classifier.train(true, "libsvm_article_not_onClassBasis_wordcount_TF_freq2_08-17_18-04");
		classifier.evaluateOnTestset("libsvm_article_not_onClassBasis_wordcount_TF_freq2_08-17_18-04");
		classifier.saveInstancesToArff("libsvm_article_not_onClassBasis_wordcount_TF_freq2_08-17_18-04" + getCurrentTimeStamp());

		//classifier.train();

		System.out.println(System.currentTimeMillis() - start);
	}

	public static String getCurrentTimeStamp() {
		return new SimpleDateFormat("MM-dd_HH-mm").format(new Date());
	}
}
