package ude.master.thesis.stance_detection.main;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;

import ude.master.thesis.stance_detection.ml.MainClassifier;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * 
 * @author Razan
 *
 */
public class StanceClassifier {

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

		/*DataSource trainDataSource = new DataSource("resources/arff_data/baseline_features08-01_02-19.arff");
		Instances trainData = trainDataSource.getDataSet();
		trainData.setClassIndex(trainData.numAttributes() - 1);
		classifier.setTrainingInstances(trainData);

		DataSource testDataSource = new DataSource("resources/arff_data/baseline_features08-01_02-19_test.arff");
		Instances testData = testDataSource.getDataSet();
		testData.setClassIndex(testData.numAttributes() - 1);
		classifier.setTestInstances(testData);*/
		
		classifier.setUseSummarization(true);

		classifier.setUseOverlapFeature(true);
		classifier.setUseRefutingFeatures(true);
		classifier.setUsePolarityFeatures(true);
		classifier.setUseBinaryCooccurraneFeatures(true);
		classifier.setUseBinaryCooccurraneStopFeatures(true);
		classifier.setUseCharGramsFeatures(true);
		classifier.setUseWordGramsFeatures(true);
		// classifier.setUseTitle(true);
		// classifier.setUseArticle(true);
		classifier.initialize();
		classifier.train(true, "libsvm_baseline_features_summarized");
		//classifier.evaluateWithCrossValidation("libsvm_baseline_features_summarized");
		classifier.evaluateOnTestset("libsvm_baseline_features_summarized");
		classifier.saveInstancesToArff("libsvm_baseline_features_summarized" + getCurrentTimeStamp());


		System.out.println(System.currentTimeMillis() - start);
	}

	public static String getCurrentTimeStamp() {
		return new SimpleDateFormat("MM-dd_HH-mm").format(new Date());
	}

}
