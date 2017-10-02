package ude.master.thesis.stance_detection.main;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.log4j.Logger;

import ude.master.thesis.stance_detection.util.FNCConstants;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.core.stopwords.WordsFromFile;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class ClassifierTools {

	final static Logger logger = Logger.getLogger(ClassifierTools.class);

	private StringToWordVector str2WordFilter;

	private Instances trainingInstances;

	private Instances testInstances;

	private Classifier classifier;

	public ClassifierTools(Instances trainingInstances, Instances testInstances, Classifier classifier) {
		this.trainingInstances = trainingInstances;
		this.testInstances = testInstances;
		this.classifier = classifier;
	}

	public AttributeSelection applyAttributSelectionFilter() {
		AttributeSelection attributeFilter = new AttributeSelection();

		ChiSquaredAttributeEval ev2 = new ChiSquaredAttributeEval(); //
		// InfoGainAttributeEval ev = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(1000);
		//ranker.setNumToSelect(70);
		ranker.setThreshold(0.0);

		attributeFilter.setSearch(ranker);
		attributeFilter.setEvaluator(ev2);

		// System.out.println(trainingInstances.toSummaryString());

		try {
			attributeFilter.setInputFormat(trainingInstances);
			System.out.println("Calculated NumToSelect:  + ranker.getCalculatedNumToSelect() + " + "from "
					+ +trainingInstances.numAttributes());
			System.out.println(trainingInstances.get(0).numAttributes());
			trainingInstances = Filter.useFilter(trainingInstances, attributeFilter);
			trainingInstances.setClass(trainingInstances.attribute("stance_class"));
			System.out.println(trainingInstances.get(0).numAttributes());
			System.out.println("**trian size = " + trainingInstances.size());

			// if (useTestset) {
			System.out.println("Calculated NumToSelect: + ranker.getCalculatedNumToSelect()  " + " from "
					+ testInstances.numAttributes());
			System.out.println(testInstances.get(0).numAttributes());
			testInstances = Filter.useFilter(testInstances, attributeFilter);
			testInstances.setClass(trainingInstances.attribute("stance_class"));
			System.out.println(testInstances.get(0).numAttributes());
			System.out.println("**test size = " + testInstances.size());
			// }

		} catch (Exception e) {
			e.printStackTrace();
		}
		return attributeFilter;
	}

	public AttributeSelection applyAttributSelectionFilter(ASEvaluation evaluator, ASSearch searcher) {
		AttributeSelection attributeFilter = new AttributeSelection();

		attributeFilter.setEvaluator(evaluator);
		attributeFilter.setSearch(searcher);

		// System.out.println(trainingInstances.toSummaryString());

		try {
			attributeFilter.setInputFormat(trainingInstances);
			System.out.println("Calculated NumToSelect from:" + trainingInstances.numAttributes());
			System.out.println(trainingInstances.get(0).numAttributes());

			trainingInstances = Filter.useFilter(trainingInstances, attributeFilter);
			trainingInstances.setClass(trainingInstances.attribute("stance_class"));

			System.out.println(trainingInstances.get(0).numAttributes());
			System.out.println("trian size = " + trainingInstances.size());

			System.out.println("Calculated NumToSelect from:  " + testInstances.numAttributes());
			System.out.println(testInstances.get(0).numAttributes());

			testInstances = Filter.useFilter(testInstances, attributeFilter);
			testInstances.setClass(trainingInstances.attribute("stance_class"));

			System.out.println(testInstances.get(0).numAttributes());
			System.out.println("test size = " + testInstances.size());

		} catch (Exception e) {
			e.printStackTrace();
		}
		return attributeFilter;
	}

	public StringToWordVector applyBoWFilter(int wordsToKeep, int minNgram, int maxNgram) {
		System.out.println("started BoW filter");
		NGramTokenizer tokenizer = new NGramTokenizer();
		// By using NGram tokenizer
		tokenizer.setNGramMinSize(minNgram);
		tokenizer.setNGramMaxSize(maxNgram);
		tokenizer.setDelimiters("[^0-9a-zA-Z]");

		str2WordFilter = new StringToWordVector();

		str2WordFilter.setTokenizer(tokenizer);
		str2WordFilter.setWordsToKeep(wordsToKeep);
		// str2WordFilter.setDoNotOperateOnPerClassBasis(true);
		str2WordFilter.setLowerCaseTokens(true);
		str2WordFilter.setMinTermFreq(1);

		// Apply Stopwordlist
		WordsFromFile stopwords = new WordsFromFile();
		stopwords.setStopwords(new File("resources/stopwords.txt"));
		str2WordFilter.setStopwordsHandler(stopwords);

		str2WordFilter.setStemmer(null);

		// Apply IDF-TF Weighting + DocLength-Normalization
		str2WordFilter.setTFTransform(false);
		str2WordFilter.setIDFTransform(false);
		// str2WordFilter.setNormalizeDocLength(
		// new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL,
		// StringToWordVector.TAGS_FILTER));

		// experimental
		str2WordFilter.setOutputWordCounts(true);

		// always first attribute
		str2WordFilter.setAttributeIndices("first");

		str2WordFilter.setAttributeNamePrefix("bow_");
		try {
			str2WordFilter.setInputFormat(trainingInstances);
			// trainingInstances.addAll(testInstances);
			trainingInstances = Filter.useFilter(trainingInstances, str2WordFilter);
			trainingInstances.setClass(trainingInstances.attribute("stance_class"));
			System.out.println("train size = " + trainingInstances.size());
			// if (useTestset) {
			testInstances = Filter.useFilter(testInstances, str2WordFilter);
			testInstances.setClass(testInstances.attribute("stance_class"));
			System.out.println("test size = " + testInstances.size());
			// }
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("finished BoW filter");
		
		return str2WordFilter;
	}

	public void evaluateWithCrossValidation(String filename) {
		// if (useTainingSet) {

		try {
			System.out.println("=== Cross Validation Evaluation ===");
			Evaluation eval = new Evaluation(trainingInstances);

			StringBuffer predsBuffer = new StringBuffer();
			CSV csv = initCSV(predsBuffer);

			long startTimeEvaluation = System.currentTimeMillis();
			eval.crossValidateModel(classifier, trainingInstances, 10, new Random(1), csv);
			long endTimeEvaluation = System.currentTimeMillis();

			System.out.println((double) (endTimeEvaluation - startTimeEvaluation) / 1000 + "s Evaluationszeit");
			logger.info("\n Evaluation with Cross Validation took: "
					+ (double) (endTimeEvaluation - startTimeEvaluation) / 1000);

			// System.out.println(eval.toSummaryString());
			// System.out.println(eval.toClassDetailsString());
			// System.out.println(trainingInstances.toSummaryString());
			// System.out.println(predsBuffer.toString());

			saveEvaluation(classifier, eval, predsBuffer, "cv" + filename + getCurrentTimeStamp(), trainingInstances);

			System.out.println("===== Evaluating on filtered (training) dataset done =====");
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Problem found when evaluating");
		}
		// }

	}

	public CSV initCSV(StringBuffer predsBuffer) {
		CSV csv = new CSV();
		csv.setHeader(trainingInstances);
		csv.setBuffer(predsBuffer);
		return csv;
	}

	private void saveEvaluation(Classifier cls, Evaluation eval, StringBuffer predsBuffer, String filename,
			Instances data) throws Exception {
		PrintWriter out = new PrintWriter(filename + ".txt");
		out.println(cls.toString()); // TODO see if it prints what it meant to
		out.println(eval.toSummaryString());
		out.println(eval.toClassDetailsString());
		out.println(eval.toMatrixString());
		out.println(data.toSummaryString());
		out.println(predsBuffer.toString());
		out.close();
	}

	private static String getCurrentTimeStamp() {
		return new SimpleDateFormat("MM-dd_HH-mm").format(new Date());
	}

	/**
	 * Training the classifier on the training data
	 */
	public void train(boolean saveModel, String modelFilename) {
		try {
			classifier.buildClassifier(trainingInstances);
			System.out.println(classifier);
			System.out.println("===== Training Finished... =====");

			if (saveModel) {
				// serialize model
				ObjectOutputStream oos = new ObjectOutputStream(
						new FileOutputStream(modelFilename + ".model"));
				oos.writeObject(classifier);
				oos.flush();
				oos.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println(e.getMessage());
		}
	}

	public void evaluateOnTestset(String resultsFilename) {
		StringBuffer predsBuffer;
		predsBuffer = new StringBuffer();
		CSV csv = initCSV(predsBuffer);
		csv.setHeader(testInstances);
		csv.setOutputFile(new File(resultsFilename));

		Evaluation eval;
		try {
			eval = new Evaluation(trainingInstances);
			eval.evaluateModel(classifier, testInstances, csv);
			saveEvaluation(classifier, eval, predsBuffer, resultsFilename + "_with_test_", testInstances);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void saveInstancesToArff(String fileName) {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainingInstances);
		
		try {
			saver.setFile(new File(ProjectPaths.ARFF_DATA_PATH + fileName + FNCConstants.TRAIN + ".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
		saver = new ArffSaver();
		saver.setInstances(testInstances);
		
		try {
			saver.setFile(new File(ProjectPaths.ARFF_DATA_PATH + fileName + FNCConstants.TEST + ".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	public static void loadData(List<List<String>> trainingStances, Map<Integer, String> trainIdBodyMap, 
			HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap,List<List<String>> testStances,
			HashMap<Integer, String> testIdBodyMap,HashMap<Integer, Map<Integer, String>> testSummIdBoyMap) throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES));

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

}
