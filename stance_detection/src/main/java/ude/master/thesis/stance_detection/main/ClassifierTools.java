package ude.master.thesis.stance_detection.main;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import org.apache.log4j.Logger;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.ChiSquaredAttributeEval;
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

	public void applyAttributSelectionFilter() {
		AttributeSelection attributeFilter = new AttributeSelection();

		/*
		 * ChiSquaredAttributeEval ev2 = new ChiSquaredAttributeEval(); //
		 * InfoGainAttributeEval ev = new InfoGainAttributeEval(); Ranker ranker
		 * = new Ranker(); // ranker.setNumToSelect(4500);
		 * ranker.setNumToSelect(1200); ranker.setThreshold(0);
		 */
		WrapperSubsetEval evaluator = new WrapperSubsetEval();
		evaluator.setClassifier(classifier);
		evaluator.setEvaluationMeasure(new SelectedTag(WrapperSubsetEval.EVAL_FMEASURE, WrapperSubsetEval.TAGS_EVALUATION));
		
		attributeFilter.setEvaluator(evaluator);
		
		BestFirst searcher = new BestFirst();
		searcher.setDirection(new SelectedTag(2, BestFirst.TAGS_SELECTION));
		attributeFilter.setSearch(searcher);
		
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

	}

	public void applyBoWFilter() {
		System.out.println("started BoW filter");
		NGramTokenizer tokenizer = new NGramTokenizer();
		// By using NGram tokenizer
		tokenizer.setNGramMinSize(1);
		tokenizer.setNGramMaxSize(2);
		tokenizer.setDelimiters("[^0-9a-zA-Z]");

		str2WordFilter = new StringToWordVector();

		str2WordFilter.setTokenizer(tokenizer);
		str2WordFilter.setWordsToKeep(1000);
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
		PrintWriter out = new PrintWriter(filename + getCurrentTimeStamp() + ".txt");
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
						new FileOutputStream(modelFilename + getCurrentTimeStamp() + ".model"));
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
		// if (useTainingSet) {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainingInstances);
		try {

			// System.out.println(trainingInstances.size());
			saver.setFile(new File("C:/arff_data/" + fileName + "_train.arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
		// }
		// if (useTestset) {
		saver = new ArffSaver();
		// System.out.println(saver);
		// System.out.println(testInstances);
		saver.setInstances(testInstances);
		try {

			// System.out.println(testInstances.size());
			saver.setFile(new File("C:/arff_data/" + fileName + "_test.arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
		// }

	}

}
