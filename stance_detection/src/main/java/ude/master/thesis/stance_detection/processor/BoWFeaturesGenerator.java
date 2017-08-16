package ude.master.thesis.stance_detection.processor;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Map;

import org.junit.runners.model.TestTimedOutException;

import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Range;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.WordsFromFile;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.AddID;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.PartitionedMultiFilter;
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class BoWFeaturesGenerator {

	// 1. concatenate every title and body pair
	// 2. create an arff of the concat string as an attribute
	// 3. apply StringToWordVector on the arff and save the vocab

	private Instances articleInstances;

	private boolean lemmatize;

	private Map<Integer, String> trainBodies;
	private List<List<String>> trainStances;

	private Map<Integer, String> testBodies;
	private List<List<String>> testStances;

	private boolean dataLoaded;

	private Instances bodiesInstances;

	private Instances titlesInstances;

	private boolean useAttributeFilter = false;

	public void process() throws IOException {
		// generateArticleInstances(true, true);
		// System.out.println("***");
		// System.out.println(articleInstances == null);
		// saveVocabularyOnDisk(articleInstances, true, 1, 1, 5000, true, 3,
		// false, true, "first");
		// generateHeadlinesAndBodiesArff(true);
		loadData(true);
		ArffLoader loader1 = new ArffLoader();
		loader1.setSource(new File("C:/arff_data/titles_08-15_13-11.arff"));
		titlesInstances = loader1.getDataSet();

		ArffLoader loader2 = new ArffLoader();
		loader2.setSource(new File("C:/arff_data/bodies_08-15_13-11.arff"));
		bodiesInstances = loader2.getDataSet();

		ArffLoader loader3 = new ArffLoader();
		loader3.setSource(new File("C:/arff_data/articles_08-15_13-10.arff"));
		articleInstances = loader3.getDataSet();

		StringToWordVector bow = getBoWFilter();

		try {
			bow.setInputFormat(articleInstances);
			articleInstances = Filter.useFilter(articleInstances, bow);
			titlesInstances = Filter.useFilter(titlesInstances, bow);
			bodiesInstances = Filter.useFilter(bodiesInstances, bow);

			titlesInstances = renameAttributes(titlesInstances, "h_");
			bodiesInstances = renameAttributes(bodiesInstances, "b_");

			saveInstancesToArff("v_article", articleInstances);
			saveInstancesToArff("v_titles", titlesInstances);
			saveInstancesToArff("v_bodies", bodiesInstances);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// vectorizeBodiesInstances();
		// vectorizeTitlesInstances();
		combineVectors();
	}

	protected Instances renameAttributes(Instances data, String prefix) throws Exception {
		Instances result;
		int i;
		ArrayList<Attribute> atts;

		// rename attributes
		atts = new ArrayList<>();
		for (i = 0; i < data.numAttributes(); i++) {
			if (i == data.classIndex())
				atts.add((Attribute) data.attribute(i).copy());
			else
				atts.add(data.attribute(i).copy(prefix + data.attribute(i).name()));
		}
		// create new dataset
		result = new Instances(data.relationName(), atts, data.numInstances());
		for (i = 0; i < data.numInstances(); i++) {
			result.add((Instance) data.instance(i).copy());
		}

		// set class if present
		if (data.classIndex() > -1)
			result.setClassIndex(data.classIndex());

		return result;
	}

	private StringToWordVector getBoWFilter() {
		NGramTokenizer tokenizer = new NGramTokenizer();
		// By using NGram tokenizer
		tokenizer.setNGramMinSize(1);
		tokenizer.setNGramMaxSize(1);
		tokenizer.setDelimiters("[^0-9a-zA-Z]");

		StringToWordVector str2WordFilter = new StringToWordVector();

		str2WordFilter.setTokenizer(tokenizer);
		str2WordFilter.setWordsToKeep(5000);
		// str2WordFilter.setDoNotOperateOnPerClassBasis(true);
		str2WordFilter.setLowerCaseTokens(true);
		str2WordFilter.setMinTermFreq(2);

		// Apply Stopwordlist
		WordsFromFile stopwords = new WordsFromFile();
		stopwords.setStopwords(new File("resources/stopwords.txt"));
		str2WordFilter.setStopwordsHandler(stopwords);

		// str2WordFilter.setStemmer(null);

		// Apply IDF-TF Weighting + DocLength-Normalization
		str2WordFilter.setTFTransform(true);
		str2WordFilter.setIDFTransform(true);
		str2WordFilter.setNormalizeDocLength(
				new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL, StringToWordVector.TAGS_FILTER));

		// experimental
		str2WordFilter.setOutputWordCounts(false);

		// always first attribute
		str2WordFilter.setAttributeIndices("first");

		return str2WordFilter;
	}

	private void combineVectors() {
		System.out.println("in combineVectors");

		System.out.println("sizes: " + titlesInstances.size() + "  " + bodiesInstances.size());

		ArrayList<Attribute> features = new ArrayList<>(Collections.list(titlesInstances.get(0).enumerateAttributes()));
		features.addAll(Collections.list(bodiesInstances.get(0).enumerateAttributes()));

		String stancesClasses[] = new String[] { "agree", "disagree", "discuss", "unrelated" };
		List<String> stanceValues = Arrays.asList(stancesClasses);
		features.add(new Attribute("stance_class", stanceValues));

		Instances train_titleBodyCombinedVecs = new Instances("fnc-1", features, trainStances.size());

		train_titleBodyCombinedVecs.setClassIndex(features.size() - 1);

		int i = 0; // trace the index of the instances
		for (List<String> s : trainStances) {
			Instance newInstance = titlesInstances.get(i).mergeInstance(bodiesInstances.get(i));
			newInstance.insertAttributeAt(newInstance.numAttributes());
			newInstance.setDataset(train_titleBodyCombinedVecs);
			System.out.println(s.get(2) + "   " + newInstance.classIndex() + "  " + newInstance.classValue() + "  "+ newInstance.classAttribute());
			//newInstance.setClassValue(s.get(2));
			newInstance.setValue(newInstance.attribute(newInstance.numAttributes()-1), s.get(2));
			System.out.println("**** "+ newInstance.classValue() + "  "+ newInstance.classAttribute());
			train_titleBodyCombinedVecs.add(newInstance);
			i++;
		}
		saveInstancesToArff("combined_train", train_titleBodyCombinedVecs);

		Instances test_titleBodyCombinedVecs = new Instances("fnc-1", features, testStances.size());

		test_titleBodyCombinedVecs.setClassIndex(features.size() - 1);

		for (List<String> s : testStances) {
			Instance newInstance = titlesInstances.get(i).mergeInstance(bodiesInstances.get(i));
			newInstance.insertAttributeAt(newInstance.numAttributes());
			newInstance.setDataset(test_titleBodyCombinedVecs);
			newInstance.setValue(newInstance.attribute(newInstance.numAttributes()-1), s.get(2));
			test_titleBodyCombinedVecs.add(newInstance);
			
			i++;
		}

		saveInstancesToArff("combined_test", test_titleBodyCombinedVecs);
		System.out.println("finished combineVectors");
	}

	private void vectorizeTitlesInstances() {
		System.out.println("in vectorizeTitlesInstances");
		NGramTokenizer tokenizer = new NGramTokenizer();
		// By using NGram tokenizer
		tokenizer.setNGramMinSize(1);
		tokenizer.setNGramMaxSize(1);
		tokenizer.setDelimiters("[^0-9a-zA-Z]");

		WordsFromFile stopwords = new WordsFromFile();
		stopwords.setStopwords(new File("resources/stopwords.txt"));

		FixedDictionaryStringToWordVector fdStr = getVectorizer(tokenizer, stopwords, false, true, true, false,
				new File("resources/dic_5000"));
		fdStr.setAttributeIndices("first");
		fdStr.setAttributeNamePrefix("h_");

		try {
			System.out.println("titles instances size = " + titlesInstances.size());
			fdStr.setInputFormat(titlesInstances);
			titlesInstances = Filter.useFilter(titlesInstances, fdStr);
			System.out.println("v titles instances size = " + titlesInstances.size());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		saveInstancesToArff("vectorized_titles", titlesInstances);
		System.out.println("finished vectorizeTitlesInstances");
	}

	private void vectorizeBodiesInstances() {
		System.out.println("in vectorizeBodiesInstances");

		NGramTokenizer tokenizer = new NGramTokenizer();
		// By using NGram tokenizer
		tokenizer.setNGramMinSize(1);
		tokenizer.setNGramMaxSize(1);
		tokenizer.setDelimiters("[^0-9a-zA-Z]");

		WordsFromFile stopwords = new WordsFromFile();
		stopwords.setStopwords(new File("resources/stopwords.txt"));

		FixedDictionaryStringToWordVector fdStr = getVectorizer(tokenizer, stopwords, false, true, true, false,
				new File("resources/dic_5000"));
		fdStr.setAttributeIndices("first");
		fdStr.setAttributeNamePrefix("b_");

		try {
			System.out.println("bodies instances size = " + titlesInstances.size());
			fdStr.setInputFormat(bodiesInstances);
			bodiesInstances = Filter.useFilter(bodiesInstances, fdStr);
			System.out.println("v bodies instances size = " + titlesInstances.size());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		saveInstancesToArff("vectorized_bodies", bodiesInstances);
		System.out.println("finished vectorizeBodiesInstances");
	}

	private FixedDictionaryStringToWordVector getVectorizer(NGramTokenizer tokenizer, WordsFromFile stopwords,
			boolean TF, boolean IDF, boolean normDoc, boolean useWordCount, File DicFile) {
		System.out.println("in getVectorizer");

		FixedDictionaryStringToWordVector fdStr2W = new FixedDictionaryStringToWordVector();

		fdStr2W.setTokenizer(tokenizer);
		fdStr2W.setLowerCaseTokens(true);

		// Apply Stopwordlist
		stopwords.setStopwords(new File("resources/stopwords.txt"));
		fdStr2W.setStopwordsHandler(stopwords);

		// Apply IDF-TF Weighting + DocLength-Normalization
		fdStr2W.setTFTransform(TF);
		fdStr2W.setIDFTransform(IDF);
		fdStr2W.setNormalizeDocLength(normDoc);

		// experimental
		fdStr2W.setOutputWordCounts(useWordCount);

		fdStr2W.setDictionaryFile(DicFile);
		System.out.println("finished getVectorizer");

		return fdStr2W;

	}

	public void generateArticleInstances(boolean withTest, boolean lemmatize) {
		System.out.println("in generateArticleInstances");

		loadData(withTest);

		List<String> headBodyArticls = new ArrayList<>();

		for (List<String> s : trainStances) {
			String article = "" + s.get(0) + " " + trainBodies.get(Integer.valueOf(s.get(1)));
			if (lemmatize)
				article = FeatureExtractor.getLemmatizedCleanStr(article);
			headBodyArticls.add(article);
		}

		if (withTest) {

			for (List<String> s : testStances) {
				String article = "" + s.get(0) + " " + testBodies.get(Integer.valueOf(s.get(1)));
				if (lemmatize)
					article = FeatureExtractor.getLemmatizedCleanStr(article);
				headBodyArticls.add(article);
			}
		}

		System.out.println("88888 ");
		System.out.println(articleInstances == null);
		articleInstances = createArticlesArffFile(headBodyArticls, "articles");
		System.out.println("finished generateArticleInstances");
		System.out.println("111 ");
		System.out.println(articleInstances == null);
		// return articleInstances;
	}

	private void loadData(boolean withTest) {
		System.out.println("in loadData");

		StanceDetectionDataReader sddr = null;
		try {
			sddr = new StanceDetectionDataReader(true, true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		trainBodies = sddr.getTrainIdBodyMap();
		trainStances = sddr.getTrainStances();

		if (withTest) {
			testBodies = sddr.getTestIdBodyMap();
			testStances = sddr.getTestStances();
		}
		dataLoaded = true;
		System.out.println("finished loadData");
	}

	public void saveVocabularyOnDisk(Instances instances, boolean useStemming, int ngram_min, int ngram_max,
			int wordsTokeep, boolean lowercase, int termFreq, boolean idf, boolean tf, String attIndices) {
		System.out.println("in saveVocabularyOnDisk");

		NGramTokenizer tokenizer = new NGramTokenizer();
		// By using NGram tokenizer
		tokenizer.setNGramMinSize(ngram_min);
		tokenizer.setNGramMaxSize(ngram_max);
		tokenizer.setDelimiters("[^0-9a-zA-Z]");

		StringToWordVector str2WordFilter = new StringToWordVector();

		str2WordFilter.setTokenizer(tokenizer);
		str2WordFilter.setWordsToKeep(wordsTokeep);
		str2WordFilter.setDoNotOperateOnPerClassBasis(true);
		str2WordFilter.setLowerCaseTokens(lowercase);
		str2WordFilter.setMinTermFreq(termFreq);

		// Apply Stopwordlist
		WordsFromFile stopwords = new WordsFromFile();
		stopwords.setStopwords(new File("resources/stopwords.txt"));
		str2WordFilter.setStopwordsHandler(stopwords);

		// Apply Stemmer
		if (useStemming) {
			SnowballStemmer stemmer = new SnowballStemmer();
			str2WordFilter.setStemmer(stemmer);
		} else
			str2WordFilter.setStemmer(null);

		// Apply IDF-TF Weighting + DocLength-Normalization
		str2WordFilter.setTFTransform(tf);
		str2WordFilter.setIDFTransform(idf);
		str2WordFilter.setNormalizeDocLength(
				new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL, StringToWordVector.TAGS_FILTER));

		// experimental
		str2WordFilter.setOutputWordCounts(true);

		// always first attribute
		str2WordFilter.setAttributeIndices(attIndices);
		str2WordFilter.setDictionaryFileToSaveTo(new File("resources/dic_5000"));
		try {
			System.out.println("ERROR");
			System.out.println(instances == null);
			str2WordFilter.setInputFormat(instances);
			instances = Filter.useFilter(instances, str2WordFilter);
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("finished saveVocabularyOnDisk");
	}

	public void generateHeadlinesAndBodiesArff(boolean withTest) {
		System.out.println("in generateHeadlinesAndBodiesArff");

		if (!dataLoaded)
			loadData(withTest);

		List<String> bodies = new ArrayList<>();

		for (List<String> s : trainStances) {
			String body = trainBodies.get(Integer.valueOf(s.get(1)));
			if (lemmatize)
				body = FeatureExtractor.getLemmatizedCleanStr(body);
			bodies.add(body);
		}

		if (withTest) {

			for (List<String> s : testStances) {
				String body = testBodies.get(Integer.valueOf(s.get(1)));
				if (lemmatize)
					body = FeatureExtractor.getLemmatizedCleanStr(body);
				bodies.add(body);
			}
		}

		bodiesInstances = createBodiesArffFile(bodies, "bodies");

		List<String> titles = new ArrayList<>();

		for (List<String> s : trainStances) {
			String title = s.get(0);
			if (lemmatize)
				title = FeatureExtractor.getLemmatizedCleanStr(title);
			titles.add(title);
		}

		if (withTest) {

			for (List<String> s : testStances) {
				String title = s.get(0);
				if (lemmatize)
					title = FeatureExtractor.getLemmatizedCleanStr(title);
				titles.add(title);
			}
		}

		titlesInstances = createTitlesArffFile(titles, "titles");

		System.out.println("finished generateHeadlinesAndBodiesArff");
	}

	private Instances createBodiesArffFile(List<String> bodies, String filename) {
		System.out.println("in createBodiesArffFile");
		ArrayList<Attribute> features = new ArrayList<>();
		features.add(new Attribute("article_Body", (List<String>) null));

		Instances instances = new Instances("bodies", features, bodies.size());

		for (String a : bodies) {
			DenseInstance instance = new DenseInstance(features.size());
			instance.setDataset(instances);
			Attribute att = instances.attribute("article_Body");
			instance.setValue(att, a);
			instances.add(instance);
		}

		/*
		 * AddID addIdFilter = new AddID();
		 * addIdFilter.setAttributeName("b_id");
		 * addIdFilter.setIDIndex("first"); try {
		 * addIdFilter.setInputFormat(instances); instances =
		 * Filter.useFilter(instances, addIdFilter); } catch (Exception e) { //
		 * TODO Auto-generated catch block e.printStackTrace(); }
		 */

		instances.setClassIndex(-1);
		saveInstancesToArff(filename, instances);
		System.out.println("finished createBodiesArffFile");

		return instances;
	}

	private Instances createTitlesArffFile(List<String> titles, String filename) {
		System.out.println("in createTitlesArffFile");

		ArrayList<Attribute> features = new ArrayList<>();
		features.add(new Attribute("article_title", (List<String>) null));

		Instances instances = new Instances("titles", features, titles.size());

		for (String t : titles) {
			DenseInstance instance = new DenseInstance(features.size());
			instance.setDataset(instances);
			Attribute att = instances.attribute("article_title");
			instance.setValue(att, t);
			instances.add(instance);
		}

		/*
		 * AddID addIdFilter = new AddID();
		 * addIdFilter.setAttributeName("t_id");
		 * addIdFilter.setIDIndex("first"); try {
		 * addIdFilter.setInputFormat(instances); instances =
		 * Filter.useFilter(instances, addIdFilter); } catch (Exception e) { //
		 * TODO Auto-generated catch block e.printStackTrace(); }
		 */

		instances.setClassIndex(-1);
		saveInstancesToArff(filename, instances);

		System.out.println("finised createTitlesArffFile");

		return instances;

	}

	private Instances createArticlesArffFile(List<String> headBodyArticls, String filename) {
		System.out.println("in createArticlesArffFile");

		ArrayList<Attribute> features = new ArrayList<>();
		features.add(new Attribute("complete_article", (List<String>) null));

		Instances instances = new Instances("articles", features, headBodyArticls.size());

		for (String a : headBodyArticls) {
			DenseInstance instance = new DenseInstance(features.size());
			instance.setDataset(instances);
			Attribute att = instances.attribute("complete_article");
			instance.setValue(att, a);
			instances.add(instance);
		}

		if (useAttributeFilter)
			applyAttributeFilter(instances);

		instances.setClassIndex(-1);
		saveInstancesToArff(filename, instances);

		System.out.println("in createArticlesArffFile");

		return instances;
	}

	private void applyAttributeFilter(Instances instances) {
		System.out.println("in applyAttributeFilter");

		AttributeSelection attributeFilter = new AttributeSelection();

		ChiSquaredAttributeEval ev2 = new ChiSquaredAttributeEval();
		// InfoGainAttributeEval ev = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		// ranker.setNumToSelect(4500);
		ranker.setNumToSelect(5000);
		ranker.setThreshold(0);

		attributeFilter.setEvaluator(ev2);
		attributeFilter.setSearch(ranker);

		try {
			attributeFilter.setInputFormat(instances);
			instances = Filter.useFilter(instances, attributeFilter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("finished applyAttributeFilter");
	}

	private void saveInstancesToArff(String filename, Instances instances) {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		try {

			// System.out.println(trainingInstances.size());
			saver.setFile(new File("C:/arff_data/" + filename + "_" + getCurrentTimeStamp() + ".arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	private String getCurrentTimeStamp() {
		return new SimpleDateFormat("MM-dd_HH-mm").format(new Date());
	}

	public Instances getArticleInstances() {
		return articleInstances;
	}

	public void setArticleInstances(Instances articleInstances) {
		this.articleInstances = articleInstances;
	}

	public boolean isUseAttributeFilter() {
		return useAttributeFilter;
	}

	public void setUseAttributeFilter(boolean useAttributeFilter) {
		this.useAttributeFilter = useAttributeFilter;
	}

	public static void main(String[] args) throws IOException {
		BoWFeaturesGenerator boWfg = new BoWFeaturesGenerator();
		boWfg.process();
	}

}
