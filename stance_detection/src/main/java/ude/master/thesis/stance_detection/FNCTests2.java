package ude.master.thesis.stance_detection;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVWriter;

import ude.master.thesis.stance_detection.main.ClassifierTools;
import ude.master.thesis.stance_detection.ml.FeaturesOrganiser2;
import ude.master.thesis.stance_detection.util.FNCConstants;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.functions.LibLINEAR;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class FNCTests2 {

	private static HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap;
	private static HashMap<Integer, Map<Integer, String>> testSummIdBoyMap;
	private static List<List<String>> trainingStances;
	private static List<List<String>> testStances;

	public static void main(String[] args) throws Exception {
		testFerreiraFeatures();
		//testRelatedUnrelatedClassifying();
		//testFullClassifier();
	}

	public static void testFullClassifier() throws Exception {
		loadData();
		// load arff data

		Instances baselineTrain = StanceDetectionDataReader
				.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "baseline_bin_new_10-21_19-13_train.arff");
		Instances baselineTest = StanceDetectionDataReader
				.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "baseline_bin_new_10-21_19-13_tset.arff");

		//Instances ferrTrain = StanceDetectionDataReader
			//	.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "ferr_BoW1000_newBparts10-03_15-02_train.arff");
		//Instances ferrTest = StanceDetectionDataReader
			//	.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "full_instances_bow1000_unlabeled_newBparts_10-03_15-02_tset.arff");

		
		//Adding neg arg
		Instances ferrTrain = StanceDetectionDataReader
				.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "ferr_BoW1000_newBparts10-25_09-05_train.arff");
		Instances ferrTest = StanceDetectionDataReader
				.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "full_instances_bow1000_unlabeled_newBparts_10-25_09-05_test.arff");

		
		// train classifier on BL features
		LibLINEAR blClassifier = new LibLINEAR();
		blClassifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));

		ClassifierTools ct1 = new ClassifierTools(baselineTrain, baselineTest, blClassifier);
		
		String time1 = FNCConstants.getCurrentTimeStamp();
		ct1.train(true, ProjectPaths.RESULTS_PATH + "baseline_bin" + time1);

		// train classifier on Ferr features
		LibLINEAR ferrClassifier = new LibLINEAR();
		ferrClassifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));

		ClassifierTools ct2 = new ClassifierTools(ferrTrain, ferrTest, ferrClassifier);

		String time2 = FNCConstants.getCurrentTimeStamp();
		// ct2.train(true, ProjectPaths.RESULTS_PATH + "Ferr_related" + time2);
		// deserialize model
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(ProjectPaths.RESULTS_PATH + "modi_ferr_1000f_newBparts_10-25_09-05.model"));
		ferrClassifier = (LibLINEAR) ois.readObject();
		ois.close();

		// classify test set and save in csv
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "Headline", "Body ID", "Stance" });

		int testSize = testStances.size();

		for (int i = 0; i < testSize; i++) {
			List<String> entry = new ArrayList<>();
			entry.add(testStances.get(i).get(0));
			entry.add(testStances.get(i).get(1));

			// 1. predict relatedness
			String relateClass = FNCConstants.BINARY_STANCE_CLASSES[(int) blClassifier
					.classifyInstance(baselineTest.get(i))];

			// 2.
			String spRelatedCls = "";
			if (relateClass.equals("related")) {
				spRelatedCls = FNCConstants.ALL_STANCE_CLASSES[(int) ferrClassifier.classifyInstance(ferrTest.get(i))];
				entry.add(spRelatedCls);
			} else {
				entry.add(relateClass);
			}
			entries.add(entry.toArray(new String[0]));
		}

		CSVWriter writer = new CSVWriter(new FileWriter(ProjectPaths.SCORING_CSV + "test_score" + time2 + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();

	}

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true, 
				ProjectPaths.TRAIN_STANCES,
				ProjectPaths.SUMMARIZED_TRAIN_BODIES2, 
				ProjectPaths.TEST_STANCESS, 
				ProjectPaths.SUMMARIZED_TEST_BODIES2);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES2));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES2));

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	public static void testRelatedUnrelatedClassifying() throws Exception {
		FeaturesOrganiser2 fo = new FeaturesOrganiser2();
		fo.loadData();

		fo.useOverlapFeature(true);
		fo.useBinaryCooccurraneFeatures(false);
		fo.useBinaryCooccurraneStopFeatures(true);
		fo.useCharGramsFeatures(true);
		fo.useWordGramsFeatures(true);
		fo.useMetricCosineSimilarity(true);
		fo.useHypernymsSimilarity(true);
		
		fo.usePPDBFeature(true);
		fo.useWord2VecAddSimilarity(false); //does not help
		fo.useLeskOverlap(true);
		fo.useTitleAndBodyParagraphVecs(false);
		
		fo.useTitleLength(true); 

		fo.useW2VMulSim(false); // does not help
		fo.useBinaryRelatedUnrelatedClasses(true);

		String filename = "baseline_bin_new_";
		fo.setArffFilename(filename);
		fo.initializeFeatures(true);

		LibLINEAR classifier = new LibLINEAR();
		classifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));

		ClassifierTools ct = new ClassifierTools(fo.getTrainingInstances(), fo.getTestInstances(), classifier);

		ct.applyAttributSelectionFilter(false, -1);

		String time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("baseline_bin_new_" + time);

		// ct.evaluateWithCrossValidation("C:/thesis_stuff/results/"+
		// "modi_ferr");
		ct.train(true, ProjectPaths.RESULTS_PATH + "baseline_bin_new_" + time);
		ct.evaluateOnTestset(ProjectPaths.RESULTS_PATH + "baseline_bin_new_" + time);
	}

	public static void testFerreiraFeatures() throws Exception {
		FeaturesOrganiser2 fo = new FeaturesOrganiser2();
		fo.loadData();

		fo.useRefutingFeatures(false);
		fo.usePolarityFeatures(false);
		
		fo.useTitleQuestionMark(true);
		fo.useTitleLength(true);
		fo.useBodyBoWCounterFeature(true);
		fo.useRootDistFeature(true);
		fo.usePPDBFeature(true);
		fo.useSVOFeature(true);//using the sum and tldr --> no change
		
		fo.useNegFeature(false);
		fo.useNegFromArguments(false);
		
		fo.useW2VMulSim(false);
		fo.usePuncCount(true);
		fo.useArgsCount(false);
		fo.useSentiments(true);
		fo.usePPDB_TLDRFeature(false);
		fo.useNegTLDRFeature(true);
		fo.useBiasCount(false);
		fo.useIDFFeature(true);
		
		fo.useWord2VecAddSimilarity(true);
		fo.useTitleAndBodyParagraphVecs(false);
		fo.useRelatedClasses(true);
		fo.useUnlabeledTestset(true);

		String filename = "ferr_BoW1000_newBparts";
		fo.setArffFilename(filename);
		fo.initializeFeatures(true);

		LibLINEAR classifier = new LibLINEAR();
		classifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));

		ClassifierTools ct = new ClassifierTools(fo.getTrainingInstances(), fo.getTestInstances(), classifier);

		StringToWordVector bow = ct.applyBoWFilter(1000, 1, 2);
		
		//StringToWordVector tfidf = ct.applyTFIDFFilter(1000, 1, 2);

		String time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("ferr_BoW1000_newBparts" + time);

		// apply to unlabled
		Instances unlabeledTestInstances = Filter.useFilter(fo.getUnlabeledTestInstances(), bow);
		 //unlabeledTestInstances = Filter.useFilter(unlabeledTestInstances, tfidf);
		// AttributeSelectioFilter properties

		//CfsSubsetEval evaluator = new CfsSubsetEval();
		//evaluator.setClassifier(classifier);
		//evaluator.setEvaluationMeasure(new SelectedTag(WrapperSubsetEval.EVAL_FMEASURE,
		//WrapperSubsetEval.TAGS_EVALUATION));

		
		//BestFirst searcher = new BestFirst(); 
		//searcher.setDirection(new SelectedTag(1, BestFirst.TAGS_SELECTION));
		//AttributeSelection attSelct = ct.applyAttributSelectionFilter(evaluator, searcher);
		 

		AttributeSelection attSelct = ct.applyAttributSelectionFilter(true, 1000);
		unlabeledTestInstances = Filter.useFilter(unlabeledTestInstances, attSelct);

		time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("ferr_BoW1000_newBparts" + time);

		ArffSaver saver = new ArffSaver();
		saver.setInstances(unlabeledTestInstances);
		try {
			saver.setFile(new File(ProjectPaths.ARFF_DATA_PATH + "full_instances_bow1000_unlabeled_newBparts_" + time
					+ FNCConstants.TEST + ".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// ct.evaluateWithCrossValidation("C:/thesis_stuff/results/"+
		// "modi_ferr");
		ct.train(true, ProjectPaths.RESULTS_PATH + "modi_ferr_1000f_newBparts_" + time);
		ct.evaluateOnTestset(ProjectPaths.RESULTS_PATH + "modi_BoW1000_1000f_newBparts_" + time);

	}

}
