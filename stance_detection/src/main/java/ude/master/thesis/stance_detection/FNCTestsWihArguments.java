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
import ude.master.thesis.stance_detection.ml.FeaturesOrganiserWithArguments;
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

public class FNCTestsWihArguments {

	public static void main(String[] args) throws Exception {
		testFerreiraFeatures();
		//testRelatedUnrelatedClassifying();
	}

	public static void testRelatedUnrelatedClassifying() throws Exception {
		FeaturesOrganiserWithArguments fo = new FeaturesOrganiserWithArguments();
		fo.loadData();

		fo.useOverlapFeature(true);
		fo.useBinaryCooccurraneFeatures(false);
		fo.useBinaryCooccurraneStopFeatures(true);
		fo.useCharGramsFeatures(true);
		fo.useWordGramsFeatures(true);
		fo.useMetricCosineSimilarity(true);
		fo.useHypernymsSimilarity(true);
		
		fo.usePPDBFeature(false);
		fo.useWord2VecAddSimilarity(true);
		fo.useLeskOverlap(false);
		fo.useTitleAndBodyParagraphVecs(false);
		
		fo.useTitleLength(false);

		fo.useBinaryRelatedUnrelatedClasses(true);

		String filename = "baseline_bin_arg_";
		fo.setArffFilename(filename);
		fo.initializeFeatures(true);

		LibLINEAR classifier = new LibLINEAR();
		classifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));

		ClassifierTools ct = new ClassifierTools(fo.getTrainingInstances(), fo.getTestInstances(), classifier);

		ct.applyAttributSelectionFilter(false, -1);

		String time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("baseline_bin_arg_" + time);

		// ct.evaluateWithCrossValidation("C:/thesis_stuff/results/"+
		// "modi_ferr");
		ct.train(true, ProjectPaths.RESULTS_PATH + "baseline_bin_arg_" + time);
		ct.evaluateOnTestset(ProjectPaths.RESULTS_PATH + "baseline_bin_arg_" + time);
	}

	public static void testFerreiraFeatures() throws Exception {
		FeaturesOrganiserWithArguments fo = new FeaturesOrganiserWithArguments();
		fo.loadData();

		fo.useRefutingFeatures(false);
		fo.usePolarityFeatures(false);
		
		fo.useTitleQuestionMark(true);
		fo.useTitleLength(false);
		
		fo.useBodyBoWCounterFeature(false);
		fo.useBodyBoWSummCounterFeature(true);
		
		fo.useRootDistFeature(true);
		fo.usePPDBFeature(true);
		fo.useSVOFeature(true);//using the sum
		fo.useNegFeature(true);
		fo.useWord2VecAddSimilarity(false);
		fo.useTitleAndBodyParagraphVecs(false);
		fo.useRelatedClasses(true);
		fo.useUnlabeledTestset(false);

		String filename = "ferr_BoW1000_args_";
		fo.setArffFilename(filename);
		fo.initializeFeatures(true);

		LibLINEAR classifier = new LibLINEAR();
		classifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));

		ClassifierTools ct = new ClassifierTools(fo.getTrainingInstances(), fo.getTestInstances(), classifier);

		StringToWordVector bow = ct.applyBoWFilter(1000, 1, 2);

		AttributeSelection attSelct = ct.applyAttributSelectionFilter(true, 1000);
		String time = FNCConstants.getCurrentTimeStamp();
		ct.saveInstancesToArff("ferr_BoW1000_args_" + time);


		// AttributeSelectioFilter properties

		//CfsSubsetEval evaluator = new CfsSubsetEval();
		//evaluator.setClassifier(classifier);
		//evaluator.setEvaluationMeasure(new SelectedTag(WrapperSubsetEval.EVAL_FMEASURE,
		//WrapperSubsetEval.TAGS_EVALUATION));

		
		//BestFirst searcher = new BestFirst(); 
		//searcher.setDirection(new SelectedTag(1, BestFirst.TAGS_SELECTION));
		//AttributeSelection attSelct = ct.applyAttributSelectionFilter(evaluator, searcher);


		// ct.evaluateWithCrossValidation("C:/thesis_stuff/results/"+
		// "modi_ferr");
		ct.train(true, ProjectPaths.RESULTS_PATH + "modi_ferr_1000f_args__" + time);
		ct.evaluateOnTestset(ProjectPaths.RESULTS_PATH + "modi_BoW1000_1000f_args__" + time);


	}

}
