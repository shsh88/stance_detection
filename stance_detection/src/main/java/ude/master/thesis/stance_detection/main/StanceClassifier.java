package ude.master.thesis.stance_detection.main;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;

import ude.master.thesis.stance_detection.ml.MainClassifier;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.classifiers.functions.SMO;

/**
 * 
 * @author Razan
 *
 */
public class StanceClassifier {

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
				new SMO());

		classifier.setUseOverlapFeature(true);
		classifier.setUseRefutingFeatures(true);
		classifier.setUsePolarityFeatures(true);
		classifier.setUseBinaryCooccurraneFeatures(true);
		classifier.setUseBinaryCooccurraneStopFeatures(true);
		classifier.setUseCharGramsFeatures(true);
		classifier.setUseWordGramsFeatures(true);
		classifier.evaluate();
		classifier.saveInstancesToArff("baseline_features" + getCurrentTimeStamp());

		//classifier.train();

		System.out.println(System.currentTimeMillis() - start);
	}

	public static String getCurrentTimeStamp() {
		return new SimpleDateFormat("MM-dd_HH-mm").format(new Date());
	}

}
