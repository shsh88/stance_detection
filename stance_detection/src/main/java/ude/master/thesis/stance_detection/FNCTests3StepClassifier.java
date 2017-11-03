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
import ude.master.thesis.stance_detection.ml.FeaturesOrganiser3StepClassifier;
import ude.master.thesis.stance_detection.ml.FeaturesOrganiser3StepClassifier;
import ude.master.thesis.stance_detection.util.FNCConstants;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class FNCTests3StepClassifier {

	private static HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap;
	private static HashMap<Integer, Map<Integer, String>> testSummIdBoyMap;
	private static List<List<String>> trainingStances;
	private static List<List<String>> testStances;

	public static void main(String[] args) throws Exception {
		testNeutralBiasedClassifier();
		// testAgreementClassifier();
		// testRelatedUnrelatedClassifying();
		// testFullClassifier();
	}

	public static void testFullClassifier() throws Exception {
		loadData();
		// load arff data

		Instances relatedTrain = StanceDetectionDataReader
				.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "baseline_bin_new_10-21_19-13_train.arff");
		Instances relatedTest = StanceDetectionDataReader
				.readInstancesFromArff(ProjectPaths.ARFF_DATA_PATH + "baseline_bin_new_10-21_19-13_tset.arff");

		Instances discussTest = StanceDetectionDataReader.readInstancesFromArff(
				ProjectPaths.ARFF_DATA_PATH + "full_instances_bow1000_unlabeled_Three_Step_10-30_15-14_test.arff");

		Instances agreeTest = StanceDetectionDataReader.readInstancesFromArff(
				ProjectPaths.ARFF_DATA_PATH + "full_instances_bow1000_unlabeled_Three_Step_agree10-30_21-27_test.arff");

		// train classifier on BL features
		LibLINEAR blClassifier = new LibLINEAR();
		blClassifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));

		ClassifierTools ct1 = new ClassifierTools(relatedTrain, relatedTest, blClassifier);

		String time1 = FNCConstants.getCurrentTimeStamp();
		ct1.train(true, ProjectPaths.RESULTS_PATH + "baseline_bin" + time1);

		// train classifier on discuss features
		LibLINEAR discussClassifier = new LibLINEAR();

		String time2 = FNCConstants.getCurrentTimeStamp();
		// deserialize model
		ObjectInputStream ois = new ObjectInputStream(
				new FileInputStream(ProjectPaths.RESULTS_PATH + "Three_Step_1000f_10-30_15-14.model"));
		discussClassifier = (LibLINEAR) ois.readObject();
		ois.close();

		// train classifier on discuss features
		LibLINEAR agreeClassifier = new LibLINEAR();

		// deserialize model
		ObjectInputStream ois2 = new ObjectInputStream(
				new FileInputStream(ProjectPaths.RESULTS_PATH + "Three_Step_1000f_agree10-30_21-27.model"));
		agreeClassifier = (LibLINEAR) ois2.readObject();
		ois2.close();

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
					.classifyInstance(relatedTest.get(i))];

			// 2.
			String spRelatedCls = "";
			if (relateClass.equals("related")) {
				spRelatedCls = FNCConstants.DISCUSS_STANCE_CLASSES[(int) discussClassifier
						.classifyInstance(discussTest.get(i))];
				if (spRelatedCls.equals("non_discuss")) {
					String agreeCls = FNCConstants.AGREE_STANCE_CLASSES[(int) agreeClassifier
							.classifyInstance(agreeTest.get(i))];
					entry.add(agreeCls);
				} else
					entry.add(spRelatedCls);
			} else {
				entry.add(relateClass);
			}
			entries.add(entry.toArray(new String[0]));
		}

		CSVWriter writer = new CSVWriter(
				new FileWriter(ProjectPaths.SCORING_CSV + "test_score_3Step_" + time2 + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();

	}

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true, ProjectPaths.TRAIN_STANCES,
				ProjectPaths.SUMMARIZED_TRAIN_BODIES2, ProjectPaths.TEST_STANCESS,
				ProjectPaths.SUMMARIZED_TEST_BODIES2);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES2));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES2));

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	public static void testRelatedUnrelatedClassifying() throws Exception {
		FeaturesOrganiser3StepClassifier fo = new FeaturesOrganiser3StepClassifier();
		fo.loadData();

		fo.useOverlapFeature(true);
		fo.useBinaryCooccurraneFeatures(false);
		fo.useBinaryCooccurraneStopFeatures(true);
		fo.useCharGramsFeatures(true);
		fo.useWordGramsFeatures(true);
		fo.useMetricCosineSimilarity(true);
		fo.useHypernymsSimilarity(true);

		fo.usePPDBFeature(true);
		fo.useWord2VecAddSimilarity(false); // does not help
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

	public static void testNeutralBiasedClassifier() throws Exception {
		FeaturesOrganiser3StepClassifier fo = new FeaturesOrganiser3StepClassifier();
		fo.loadData();

		fo.useRefutingFeatures(false);
		fo.usePolarityFeatures(false);

		fo.useTitleQuestionMark(true);
		fo.useTitleLength(false);
		fo.useBodyBoWCounterFeature(true);
		fo.useRootDistFeature(true);
		fo.usePPDBFeature(false);
		fo.useSVOFeature(false);// using the sum and tldr --> no change

		fo.useNegFeature(false);
		fo.useNegFromArguments(false);

		fo.useW2VMulSim(false); // no benefit
		fo.usePuncCount(true);
		fo.useArgsCount(false);
		fo.useSentiments(true);
		fo.usePPDB_TLDRFeature(false);
		fo.useNegTLDRFeature(false);
		fo.useBiasCount(true);
		fo.useSentenceLength(true);
		fo.usePOSTags(false);

		fo.useWord2VecAddSimilarity(true);
		fo.useTitleAndBodyParagraphVecs(true);
		fo.useDiscussClasses(true);
		fo.useUnlabeledTestset(true);

		String filename = "Three_Step_BoW1000_";
		fo.setArffFilename(filename);
		fo.initializeFeatures(true);

		// CostSensitiveClassifier csc = new CostSensitiveClassifier();
		// csc.setOptions(weka.core.Utils.splitOptions("-cost-matrix \"[0.0 3.0;
		// 5.0 0.0]\" -S 1 -W weka.classifiers.functions.LibLINEAR -- -S 6 -C
		// 1.0 -E 0.001 -B 1.0"));
		LibLINEAR classifier = new LibLINEAR();
		classifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));

		ClassifierTools ct = new ClassifierTools(fo.getTrainingInstances(), fo.getTestInstances(), classifier);
		StringToWordVector bow = null;
		if (fo.isBodyBoWCounterFeatureUsed())
			bow = ct.applyBoWFilter(500, 1, 2);

		StringToWordVector pos_bow = null;
		if (fo.isPOSTagsUse()) {
			pos_bow = ct.applyPOSBoWFilter(500, 1, 1);

		}

		String time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("Three_Step_BoW1000" + time);

		Instances unlabeledTestInstances = fo.getUnlabeledTestInstances();
		// apply to unlabled
		if (fo.isBodyBoWCounterFeatureUsed())
			unlabeledTestInstances = Filter.useFilter(unlabeledTestInstances, bow);

		if (fo.isPOSTagsUse()) {
			unlabeledTestInstances = Filter.useFilter(unlabeledTestInstances, pos_bow);
		}

		// AttributeSelectioFilter properties

		// CfsSubsetEval evaluator = new CfsSubsetEval();
		// evaluator.setClassifier(classifier);
		// evaluator.setEvaluationMeasure(new
		// SelectedTag(WrapperSubsetEval.EVAL_FMEASURE,
		// WrapperSubsetEval.TAGS_EVALUATION));

		// BestFirst searcher = new BestFirst();
		// searcher.setDirection(new SelectedTag(1, BestFirst.TAGS_SELECTION));
		// AttributeSelection attSelct =
		// ct.applyAttributSelectionFilter(evaluator, searcher);

		AttributeSelection attSelct = ct.applyAttributSelectionFilter(true, 500);
		unlabeledTestInstances = Filter.useFilter(unlabeledTestInstances, attSelct);

		time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("Three_Step_BoW1000_" + time);

		ArffSaver saver = new ArffSaver();
		saver.setInstances(unlabeledTestInstances);
		try {
			saver.setFile(new File(ProjectPaths.ARFF_DATA_PATH + "full_instances_bow1000_unlabeled_Three_Step_" + time
					+ FNCConstants.TEST + ".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// ct.evaluateWithCrossValidation("C:/thesis_stuff/results/"+
		// "modi_ferr");
		ct.train(true, ProjectPaths.RESULTS_PATH + "Three_Step_1000f_" + time);
		ct.evaluateOnTestset(ProjectPaths.RESULTS_PATH + "Three_Step_BoW1000_1000f_" + time);

	}

	public static void testAgreementClassifier() throws Exception {
		FeaturesOrganiser3StepClassifier fo = new FeaturesOrganiser3StepClassifier();
		fo.loadData();

		fo.useRefutingFeatures(false);
		fo.usePolarityFeatures(false);

		fo.useTitleQuestionMark(false);
		fo.useTitleLength(false);
		fo.useBodyBoWCounterFeature(true);
		fo.useRootDistFeature(true);
		fo.usePPDBFeature(false);
		fo.useSVOFeature(true);// using the sum and tldr --> no change

		fo.useNegFeature(false);
		fo.useNegFromArguments(false);

		fo.useW2VMulSim(false);
		fo.usePuncCount(true);
		fo.useArgsCount(true);
		fo.useSentiments(true);
		fo.usePPDB_TLDRFeature(true);
		fo.useNegTLDRFeature(true);
		fo.useBiasCount(true);

		fo.useWord2VecAddSimilarity(true);
		fo.useTitleAndBodyParagraphVecs(true);
		fo.useAgreeClasses(true);
		fo.useUnlabeledTestset(true);

		String filename = "Three_Step_BoW1000_agree";
		fo.setArffFilename(filename);
		fo.initializeFeatures(true);

		LibLINEAR classifier = new LibLINEAR();
		classifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));

		ClassifierTools ct = new ClassifierTools(fo.getTrainingInstances(), fo.getTestInstances(), classifier);

		StringToWordVector bow = ct.applyBoWFilter(500, 1, 2);

		String time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("Three_Step_BoW1000_agree" + time);

		// apply to unlabled
		Instances unlabeledTestInstances = Filter.useFilter(fo.getUnlabeledTestInstances(), bow);

		// AttributeSelectioFilter properties

		// CfsSubsetEval evaluator = new CfsSubsetEval();
		// evaluator.setClassifier(classifier);
		// evaluator.setEvaluationMeasure(new
		// SelectedTag(WrapperSubsetEval.EVAL_FMEASURE,
		// WrapperSubsetEval.TAGS_EVALUATION));

		// BestFirst searcher = new BestFirst();
		// searcher.setDirection(new SelectedTag(1, BestFirst.TAGS_SELECTION));
		// AttributeSelection attSelct =
		// ct.applyAttributSelectionFilter(evaluator, searcher);

		AttributeSelection attSelct = ct.applyAttributSelectionFilter(true, 500);
		unlabeledTestInstances = Filter.useFilter(unlabeledTestInstances, attSelct);

		time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("Three_Step_BoW1000_" + time);

		ArffSaver saver = new ArffSaver();
		saver.setInstances(unlabeledTestInstances);
		try {
			saver.setFile(new File(ProjectPaths.ARFF_DATA_PATH + "full_instances_bow1000_unlabeled_Three_Step_agree"
					+ time + FNCConstants.TEST + ".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// ct.evaluateWithCrossValidation("C:/thesis_stuff/results/"+
		// "modi_ferr");
		ct.train(true, ProjectPaths.RESULTS_PATH + "Three_Step_1000f_agree" + time);
		ct.evaluateOnTestset(ProjectPaths.RESULTS_PATH + "Three_Step_BoW1000_1000f_agree" + time);

	}

}
