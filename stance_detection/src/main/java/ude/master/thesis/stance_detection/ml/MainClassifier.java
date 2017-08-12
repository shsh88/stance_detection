package ude.master.thesis.stance_detection.ml;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.log4j.Logger;
import org.apache.lucene.search.UsageTrackingQueryCachingPolicy;

import ude.master.thesis.stance_detection.processor.FeatureExtractor;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Range;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.WordsFromFile;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

import weka.attributeSelection.ChiSquaredAttributeEval;

public class MainClassifier {

	final static Logger logger = Logger.getLogger(MainClassifier.class);

	private static final String RELATION_NAME = "fnc-1";
	// Baseline features settings
	// TODO: split the feature types in a better way (like in hs project)
	private boolean useOverlapFeature = false;
	private boolean useRefutingFeatures = false;
	private boolean usePolarityFeatures = false;
	private boolean useBinaryCooccurraneFeatures = false;
	private boolean useBinaryCooccurraneStopFeatures = false; // Same as binary
	// cooccurrence but
	// ignores stopwords.
	private boolean useCharGramsFeatures = false;
	private boolean useWordGramsFeatures = false;

	private boolean useTitle = false;
	private boolean useArticle = false;

	private boolean useTitleEmbedding = false;
	private boolean useArticleEmbedding = false;

	private Map<Integer, String> trainingIdBodyMap;
	private List<List<String>> trainingStances;
	private Instances trainingInstances;

	private boolean useTestset = false;
	private boolean useTainingSet = false;

	private boolean useAttributeSelectionFilter = false;

	private Map<Integer, String> testIdBodyMap;
	private List<List<String>> testStances;
	private Instances testInstances;

	private StringToWordVector str2WordFilter;

	private int textNGramMinSize = 1;
	private int textNGramMaxSize = 1;

	private Classifier classifier;

	private AttributeSelection attributeFilter;

	private boolean evaluate = true;

	public MainClassifier(Map<Integer, String> trainingIdBodyMap, List<List<String>> trainingStances,
			Classifier classifier) {
		this.trainingIdBodyMap = trainingIdBodyMap;
		this.trainingStances = trainingStances;

		this.classifier = classifier;
	}

	public MainClassifier(Map<Integer, String> trainingIdBodyMap, List<List<String>> trainingStances,
			Map<Integer, String> testIdBodyMap, List<List<String>> testStances, Classifier classifier) {
		this.trainingIdBodyMap = trainingIdBodyMap;
		this.trainingStances = trainingStances;

		this.useTestset = true;
		this.testIdBodyMap = testIdBodyMap;
		this.testStances = testStances;

		this.useTainingSet = true;
		this.classifier = classifier;
	}

	private void init() {

		// TODO here we also build embeddings
		if (useTainingSet) {
			trainingInstances = initializeInstances("fnc-1", trainingStances, trainingIdBodyMap);

			trainingInstances.setClassIndex(trainingInstances.numAttributes() - 1);
		}

		if (useTestset) {
			testInstances = initializeInstances("fnc-1", testStances, testIdBodyMap);

			testInstances.setClassIndex(testInstances.numAttributes() - 1);
		}

		if (useTitle || useArticle) {
			initBoWFilter();
		}

		if (useAttributeSelectionFilter) {
			applyAttributSelectionFilter();
		}
	}

	private void applyAttributSelectionFilter() {
		attributeFilter = new AttributeSelection();
		
		ChiSquaredAttributeEval ev2 = new ChiSquaredAttributeEval();
		// InfoGainAttributeEval ev = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		// ranker.setNumToSelect(4500);
		ranker.setNumToSelect(5000);
		ranker.setThreshold(0);

		attributeFilter.setEvaluator(ev2);
		attributeFilter.setSearch(ranker);

		try {
			attributeFilter.setInputFormat(trainingInstances);
			System.out.println("Calculated NumToSelect:  + ranker.getCalculatedNumToSelect() + " + "from "
					+ +trainingInstances.numAttributes());
			System.out.println(trainingInstances.get(0).numAttributes());
			trainingInstances = Filter.useFilter(trainingInstances, attributeFilter);
			System.out.println(trainingInstances.get(0).numAttributes());
			System.out.println("**trian size = " + trainingInstances.size());

			if (useTestset) {
				System.out.println("Calculated NumToSelect: + ranker.getCalculatedNumToSelect()  " + " from "
						+ testInstances.numAttributes());
				System.out.println(testInstances.get(0).numAttributes());
				testInstances = Filter.useFilter(testInstances, attributeFilter);
				System.out.println(testInstances.get(0).numAttributes());
				System.out.println("**test size = " + testInstances.size());
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	private void initBoWFilter() {
		NGramTokenizer tokenizer = new NGramTokenizer();
		System.out.println("Heeeeere");
		// By using NGram tokenizer
		tokenizer.setNGramMinSize(textNGramMinSize);
		tokenizer.setNGramMaxSize(textNGramMaxSize);
		tokenizer.setDelimiters("[^0-9a-zA-Z]");

		str2WordFilter = new StringToWordVector();

		str2WordFilter.setTokenizer(tokenizer);
		str2WordFilter.setWordsToKeep(1000000);
		// str2WordFilter.setDoNotOperateOnPerClassBasis(true);
		str2WordFilter.setLowerCaseTokens(true);
		str2WordFilter.setMinTermFreq(3);

		// Apply Stopwordlist
		WordsFromFile stopwords = new WordsFromFile();
		stopwords.setStopwords(new File("resources/stopwords.txt"));
		str2WordFilter.setStopwordsHandler(stopwords);

		// Apply Stemmer
		SnowballStemmer stemmer = new SnowballStemmer();
		str2WordFilter.setStemmer(stemmer);

		// Apply IDF-TF Weighting + DocLength-Normalization
		str2WordFilter.setTFTransform(true);
		str2WordFilter.setIDFTransform(true);
		str2WordFilter.setNormalizeDocLength(
				new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL, StringToWordVector.TAGS_FILTER));

		// experimental
		str2WordFilter.setOutputWordCounts(true);

		// always first attribute
		str2WordFilter.setAttributeIndices("first,2");
		try {
			str2WordFilter.setInputFormat(trainingInstances);
			// trainingInstances.addAll(testInstances);
			trainingInstances = Filter.useFilter(trainingInstances, str2WordFilter);
			System.out.println("train size = " + trainingInstances.size());
			if (useTestset) {
				testInstances = Filter.useFilter(testInstances, str2WordFilter);
				System.out.println("test size = " + testInstances.size());
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	private Instances initializeInstances(String relationName, List<List<String>> stances,
			Map<Integer, String> idBodyMap) {
		ArrayList<Attribute> features = new ArrayList<>();

		/**
		 * TODO: When using BOW filter: 1. we add the text value as is as a
		 * seprate attribute 2. apply the StringToWord vector on those attribute
		 * 3. see initializeBOWFilter() in hs and useMessage in
		 * initializeInstances(...)
		 */
		if (useTitle) {
			features.add(new Attribute("title_head", (List<String>) null));
		}
		if (useArticle) {
			features.add(new Attribute("article_body", (List<String>) null));
		}
		if (useOverlapFeature) {
			features.add(new Attribute("word_overlap"));
		}

		if (useRefutingFeatures) {
			for (String refute : FeatureExtractor.refutingWords) {
				features.add(new Attribute("refute_" + refute));
			}
		}

		if (usePolarityFeatures) {
			features.add(new Attribute("pol_head"));
			features.add(new Attribute("pol_body"));
		}

		if (useBinaryCooccurraneFeatures) {
			features.add(new Attribute("bin_co_occ_count"));
			features.add(new Attribute("bin_co_occ_255")); // just in the first
															// 255 words
		}

		if (useBinaryCooccurraneStopFeatures) {
			features.add(new Attribute("bin_co_occ_stop_count"));
			features.add(new Attribute("bin_co_occ_stop_255"));
		}

		if (useCharGramsFeatures) {
			int[] cgramSizes = { 2, 8, 4, 16 };
			for (int size : cgramSizes) {
				features.add(new Attribute("cgram_hits_" + size));
				features.add(new Attribute("cgram_early_hits_" + size));
				features.add(new Attribute("cgram_first_hits_" + size));
				// TODO: Not in baseline
				// features.add(new Attribute("cgram_tail_hits_" + size));
			}
		}

		if (useWordGramsFeatures) {
			int[] ngramSizes = { 2, 3, 4, 5, 6 };
			for (int size : ngramSizes) {
				features.add(new Attribute("ngram_hits_" + size));
				features.add(new Attribute("ngram_early_hits_" + size));
				// TODO: Not in baseline
				// features.add(new Attribute("ngram_tail_hits" + size));
			}
		}

		// Add the classs attribute
		String stancesClasses[] = new String[] { "agree", "disagree", "discuss", "unrelated" };
		List<String> stanceValues = Arrays.asList(stancesClasses);
		features.add(new Attribute("stance_class", stanceValues));

		Instances instances = new Instances(RELATION_NAME, features, stances.size());

		instances.setClassIndex(features.size() - 1);

		assignFeaturesValues(stances, idBodyMap, instances, features.size());
		return instances;
	}

	private void assignFeaturesValues(List<List<String>> stances, Map<Integer, String> idBodyMap, Instances instances,
			int featuresSize) {
		System.out.println("Started getting instances");
		int i = 0;
		for (List<String> stance : stances) {
			String headline = stance.get(0);
			String body = idBodyMap.get(Integer.valueOf(stance.get(1)));

			DenseInstance instance = createInstance(headline, body, instances, featuresSize);
			// System.out.println(stance.get(2));
			instance.setClassValue(stance.get(2));
			instances.add(instance);

			i++;
			//if (i == 20)
			//break;
			if (i % 10000 == 0)
				System.out.println("Have read " + instances.size() + " instances");
		}

		System.out.println("Finished getting instances");

	}

	/**
	 * Converts the title and body pair to features and wrap them in an Instance
	 * object
	 * 
	 * @param headline
	 * @param body
	 * @param instances
	 * @param featuresSize
	 * @return
	 */
	private DenseInstance createInstance(String headline, String body, Instances instances, int featuresSize) {
		// Create instance and set the number of features
		DenseInstance instance = new DenseInstance(featuresSize);

		instance.setDataset(instances);

		if (useTitle) {
			Attribute titleAtt = instances.attribute("title_head");
			instance.setValue(titleAtt, headline);
		}

		if (useArticle) {
			Attribute articleAtt = instances.attribute("article_body");
			instance.setValue(articleAtt, body);
		}

		if (useOverlapFeature) {
			Attribute wordOverlapAtt = instances.attribute("word_overlap");
			instance.setValue(wordOverlapAtt, FeatureExtractor.getWordOverlapFeature(headline, body));
		}

		if (useRefutingFeatures) {
			for (String refute : FeatureExtractor.refutingWords) {
				Attribute refuteAtts = instances.attribute("refute_" + refute);
				instance.setValue(refuteAtts, FeatureExtractor.getRefutingFeature(headline, refute));
			}
		}

		// TODO: split to 2 features use
		if (usePolarityFeatures) {
			Attribute headPolarityAtt = instances.attribute("pol_head");
			instance.setValue(headPolarityAtt, FeatureExtractor.calculatePolarity(headline));

			Attribute bodyPolarityAtt = instances.attribute("pol_body");
			instance.setValue(bodyPolarityAtt, FeatureExtractor.calculatePolarity(body));
		}
		// TODO: split to 2 features use
		if (useBinaryCooccurraneFeatures) {
			Attribute binCoOccAtt = instances.attribute("bin_co_occ_count");
			instance.setValue(binCoOccAtt, FeatureExtractor.getBinaryCoOccurenceFeatures(headline, body).get(0));

			Attribute binCoOccEarlyAtt = instances.attribute("bin_co_occ_255");
			instance.setValue(binCoOccEarlyAtt, FeatureExtractor.getBinaryCoOccurenceFeatures(headline, body).get(1));
		}

		// TODO: split to 2 features use
		if (useBinaryCooccurraneStopFeatures) {
			List<Integer> f = FeatureExtractor.getBinaryCoOccurenceStopFeatures(headline, body);
			Attribute binCoOccAtt = instances.attribute("bin_co_occ_stop_count");
			instance.setValue(binCoOccAtt, f.get(0));

			Attribute binCoOccEarlyAtt = instances.attribute("bin_co_occ_stop_255");
			instance.setValue(binCoOccEarlyAtt, f.get(1));
		}

		String cleanHeadline = FeatureExtractor.clean(headline);
		String cleanBody = FeatureExtractor.clean(body);

		// TODO: split to 3 features use
		if (useCharGramsFeatures) {
			int[] cgramSizes = { 2, 8, 4, 16 };
			for (int size : cgramSizes) {
				List<Integer> f = FeatureExtractor.getCharGramsFeatures(cleanHeadline, cleanBody, size);

				Attribute cgramHitsAtt = instances.attribute("cgram_hits_" + size);
				instance.setValue(cgramHitsAtt, f.get(0));

				Attribute cgramEarlyHitsAtt = instances.attribute("cgram_early_hits_" + size);
				instance.setValue(cgramEarlyHitsAtt, f.get(1));

				Attribute cgramFirstHitsAtt = instances.attribute("cgram_first_hits_" + size);
				instance.setValue(cgramFirstHitsAtt, f.get(2));
				// TODO: Not in baseline
				// features.add(new Attribute("cgram_tail_hits_" + size));
			}

		}

		if (useWordGramsFeatures) {
			int[] ngramSizes = { 2, 3, 4, 5, 6 };
			for (int size : ngramSizes) {
				List<Integer> f = FeatureExtractor.getNGramsFeatures(cleanHeadline, cleanBody, size);

				Attribute ngramHitAtt = instances.attribute("ngram_hits_" + size);
				instance.setValue(ngramHitAtt, f.get(0));

				Attribute ngramEarlyHitsAtt = instances.attribute("ngram_early_hits_" + size);
				instance.setValue(ngramEarlyHitsAtt, f.get(1));
			}
		}

		return instance;
	}

	public boolean isUseOverlapFeature() {
		return useOverlapFeature;
	}

	public void setUseOverlapFeature(boolean useOverlapFeature) {
		this.useOverlapFeature = useOverlapFeature;
	}

	public boolean isUseRefutingFeatures() {
		return useRefutingFeatures;
	}

	public void setUseRefutingFeatures(boolean useRefutingFeatures) {
		this.useRefutingFeatures = useRefutingFeatures;
	}

	public boolean isUsePolarityFeatures() {
		return usePolarityFeatures;
	}

	public void setUsePolarityFeatures(boolean usePolarityFeatures) {
		this.usePolarityFeatures = usePolarityFeatures;
	}

	public boolean isUseBinaryCooccurraneFeatures() {
		return useBinaryCooccurraneFeatures;
	}

	public void setUseBinaryCooccurraneFeatures(boolean useBinaryCooccurraneFeatures) {
		this.useBinaryCooccurraneFeatures = useBinaryCooccurraneFeatures;
	}

	public boolean isUseCharGramsFeatures() {
		return useCharGramsFeatures;
	}

	public void setUseCharGramsFeatures(boolean useCharGramsFeatures) {
		this.useCharGramsFeatures = useCharGramsFeatures;
	}

	public boolean isUseWordGramsFeatures() {
		return useWordGramsFeatures;
	}

	public void setUseWordGramsFeatures(boolean useWordGramsFeatures) {
		this.useWordGramsFeatures = useWordGramsFeatures;
	}

	public boolean isUseTitle() {
		return useTitle;
	}

	public void setUseTitle(boolean useTitle) {
		this.useTitle = useTitle;
	}

	public boolean isUseArticle() {
		return useArticle;
	}

	public void setUseArticle(boolean useArticle) {
		this.useArticle = useArticle;
	}

	public boolean isUseBinaryCooccurraneStopFeatures() {
		return useBinaryCooccurraneStopFeatures;
	}

	public void setUseBinaryCooccurraneStopFeatures(boolean useBinaryCooccurraneStopFeatures) {
		this.useBinaryCooccurraneStopFeatures = useBinaryCooccurraneStopFeatures;
	}

	public boolean isUseAttributeSelectionFilter() {
		return useAttributeSelectionFilter;
	}

	public void setUseAttributeSelectionFilter(boolean useAttributeSelectionFilter) {
		this.useAttributeSelectionFilter = useAttributeSelectionFilter;
	}

	public boolean isUseTitleEmbedding() {
		return useTitleEmbedding;
	}

	public void setUseTitleEmbedding(boolean useTitleEmbedding) {
		this.useTitleEmbedding = useTitleEmbedding;
	}

	public boolean isUseArticleEmbedding() {
		return useArticleEmbedding;
	}

	public void setUseArticleEmbedding(boolean useArticleEmbedding) {
		this.useArticleEmbedding = useArticleEmbedding;
	}

	public boolean isEvaluate() {
		return evaluate;
	}

	public void setEvaluate(boolean evaluate) {
		this.evaluate = evaluate;
	}

	public Instances getTrainingInstances() {
		return trainingInstances;
	}

	public void setTrainingInstances(Instances trainingInstances) {
		this.trainingInstances = trainingInstances;
	}

	public Instances getTestInstances() {
		return testInstances;
	}

	public void setTestInstances(Instances testInstances) {
		this.testInstances = testInstances;
	}

	public void evaluate() {
		if (useTainingSet) {
			if (trainingInstances == null) {
				long startTimeExtraction = System.currentTimeMillis();
				init();
				long endTimeExtraction = System.currentTimeMillis();
				System.out.println((double) (endTimeExtraction - startTimeExtraction) / 1000 + "s Feature-Extraktion");
				logger.info(
						"\n Feature-Extraktionszeit(s): " + (double) (endTimeExtraction - startTimeExtraction) / 1000);

			}
			if (evaluate)
				try {
					System.out.println("=== Evaluation ===");
					Evaluation eval = new Evaluation(trainingInstances);
					StringBuffer predsBuffer = new StringBuffer();
					CSV csv = new CSV();
					csv.setHeader(trainingInstances);
					csv.setBuffer(predsBuffer);
					long startTimeEvaluation = System.currentTimeMillis();
					eval.crossValidateModel(classifier, trainingInstances, 10, new Random(1), csv);
					long endTimeEvaluation = System.currentTimeMillis();

					System.out.println((double) (endTimeEvaluation - startTimeEvaluation) / 1000 + "s Evaluationszeit");
					logger.info("\n Evaluationzeit(s): " + (double) (endTimeEvaluation - startTimeEvaluation) / 1000);

					System.out.println(eval.toSummaryString());
					System.out.println(eval.toClassDetailsString());
					System.out.println(trainingInstances.toSummaryString());
					//System.out.println(predsBuffer.toString());

					// classifier.classifyInstance(instance)
					saveEvaluation(eval, predsBuffer, "_without_test_", trainingInstances);

					predsBuffer = new StringBuffer();
					csv = new CSV();
					csv.setHeader(testInstances);
					csv.setBuffer(predsBuffer);
					classifier.buildClassifier(trainingInstances);
					eval.evaluateModel(classifier, testInstances, csv);
					saveEvaluation(eval, predsBuffer, "_with_test_", testInstances);

					// serialize model
					 ObjectOutputStream oos = new ObjectOutputStream(
					                            new FileOutputStream("resources/models/libsvm_"+ getCurrentTimeStamp()+ ".model"));
					 oos.writeObject(classifier);
					 oos.flush();
					 oos.close();
					System.out.println("===== Evaluating on filtered (training) dataset done =====");
				} catch (Exception e) {
					e.printStackTrace();
					System.out.println("Problem found when evaluating");
				}
		}

	}

	private void saveEvaluation(Evaluation eval, StringBuffer predsBuffer, String addName, Instances data)
			throws Exception {
		PrintWriter out = new PrintWriter(
				"resources/arff_data/" + "evaluation_svm_cv_BoW_" + addName + getCurrentTimeStamp() + ".txt");
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
	public void train() {
		try {

			if (trainingInstances == null) {
				init();
			}

			classifier.buildClassifier(trainingInstances);
			System.out.println(classifier);
			System.out.println("===== Training Finished... =====");
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println(e.getMessage());
		}
	}

	public void saveInstancesToArff(String fileName) {
		if (useTainingSet) {
			ArffSaver saver = new ArffSaver();
			saver.setInstances(trainingInstances);
			try {

				// System.out.println(trainingInstances.size());
				saver.setFile(new File("resources/arff_data/" + fileName + ".arff"));
				// saver.setDestination(new File("./data/test.arff")); //
				// **not**
				// necessary in 3.5.4 and later
				saver.writeBatch();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		if (useTestset) {
			ArffSaver saver = new ArffSaver();
			// System.out.println(saver);
			// System.out.println(testInstances);
			saver.setInstances(testInstances);
			try {

				// System.out.println(testInstances.size());
				saver.setFile(new File("resources/arff_data/" + fileName + "_test.arff"));
				saver.writeBatch();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

	}

}
