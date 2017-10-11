package ude.master.thesis.stance_detection.processor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.ISynsetID;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import edu.mit.jwi.item.Pointer;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.POSTaggerAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.BodySummerizer2;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class HypernymSimilarityWithArguments {
	protected StanfordCoreNLP pipeline;
	public static final String VERB = "verb";
	public static final String NOUN = "noun";
	public static final String ADJ = "Adjective";

	private static Word2Vec googleVec;

	private static FileHashMap<Integer, Map<String, Set<String>>> bodiesPosMap;
	private static FileHashMap<String, Map<String, Set<String>>> titlesPosMap;

	private static List<List<String>> trainingStances = new ArrayList<>();
	private IDictionary dict;
	private static List<List<String>> testStances = new ArrayList<List<String>>();
	private static Map<Integer, String> trainingIdBoyMap = new HashMap<>();
	private static HashMap<Integer, String> testIdBoyMap = new HashMap<>();

	public HypernymSimilarityWithArguments() {
		// Create StanfordCoreNLP object properties, with POS tagging
		// (required for lemmatization), and lemmatization
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma");

		// StanfordCoreNLP loads a lot of models, so you probably
		// only want to do this once per execution
		this.pipeline = new StanfordCoreNLP(props);
	}

	public Map<String, Set<String>> getPosMap(String documentText) {
		Map<String, Set<String>> posMap = new HashMap<>();
		posMap.put(VERB, new HashSet<String>());
		posMap.put(NOUN, new HashSet<String>());
		posMap.put(ADJ, new HashSet<String>());
		// List<String> lemmas = new LinkedList<String>();
		// List<String> poss = new LinkedList<String>();

		// create an empty Annotation just with the given text
		Annotation document = new Annotation(documentText);

		// run all Annotators on this text
		this.pipeline.annotate(document);

		// Iterate over all of the sentences found
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		for (CoreMap sentence : sentences) {
			// Iterate over all tokens in a sentence
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// Retrieve and add the lemma for each word into the list of
				// lemmas
				// lemmas.add(token.get(LemmaAnnotation.class));
				// poss.add(token.get(PartOfSpeechAnnotation.class));
				String pos = token.get(PartOfSpeechAnnotation.class);
				if (pos.startsWith("VB"))
					posMap.get(VERB).add(token.get(LemmaAnnotation.class));
				if (pos.startsWith("NN"))
					posMap.get(NOUN).add(token.get(LemmaAnnotation.class));
				if (pos.startsWith("JJ"))
					posMap.get(ADJ).add(token.get(LemmaAnnotation.class));

			}
		}
		// System.out.println(poss.toString());
		return posMap;
	}
	// 1. we need to get the nouns / verbs / adjectives from the data

	public void saveBodiesPOSTagMapFile(HashMap<Integer, String> summIdBoyMap, String filepath) throws Exception {

		// HashMap to save values
		FileHashMap<Integer, Map<String, Set<String>>> posMap = new FileHashMap<Integer, Map<String, Set<String>>>(
				filepath, FileHashMap.FORCE_OVERWRITE);

		for (Entry<Integer, String> bodyParts : summIdBoyMap.entrySet()) {

			Map<String, Set<String>> fAll = getPOSMapAll(bodyParts.getValue());

			posMap.put(bodyParts.getKey(), fAll);

		}

		posMap.save();
		posMap.close();

		System.out.println("saved saved saved");

	}

	public void saveTitlePOSTagMapFile(Set<String> titles, String filepath) throws Exception {

		// HashMap to save values
		FileHashMap<String, Map<String, Set<String>>> posMap = new FileHashMap<String, Map<String, Set<String>>>(
				filepath, FileHashMap.FORCE_OVERWRITE);

		for (String t : titles) {

			posMap.put(t, getPOSMapAll(t));

		}

		posMap.save();
		posMap.close();

		System.out.println("saved saved saved");

	}

	private Map<String, Set<String>> getPOSMapAll(String text) {
		return getPosMap(text);
	}

	public void getHypSimFeatureVector(List<List<String>> stances, Map<Integer, String> trainingIdBoyMap,
			String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, Double> hypSims = new FileHashMap<String, Double>(filepath, FileHashMap.FORCE_OVERWRITE);

		if ((bodiesPosMap == null) && (titlesPosMap == null)) {
			bodiesPosMap = loadBodiesPosAsHashFile(ProjectPaths.BODIES_ARGS_POS_MAP_PATH);
			titlesPosMap = loadtitlesPosAsHashFile(ProjectPaths.TITLES_POS_MAP_PATH);
		}

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		csvHeader.add("hyp_sim");

		entries.add(csvHeader.toArray(new String[0]));

		int k = 0;
		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			int bodyId = Integer.valueOf(s.get(1));

			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			String body = trainingIdBoyMap.get(bodyId);

			Double d = getHypSimfeatureBody(title, body,bodyId);

			entry.add(String.valueOf(d));

			// System.out.println(entry.size());

			if (entry.size() != 4)
				throw new Exception("not 1 features");

			hypSims.put(title + s.get(1), d);

			entries.add(entry.toArray(new String[0]));

			k++;
			if (k % 1000 == 0)
				System.out.println(k + " processed");
		}

		hypSims.save();
		hypSims.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private Double getHypSimfeatureBody(String title, String body, int bodyId) throws IOException {

		if (body.equals("")) {// Any way we are not going to use empty bodies in
								// training and testing
			return -100.0;
		}

		Double f = calcHypSim(title, bodyId);
		return f;
	}

	private Double calcHypSim(String title, int bodyId) throws IOException {

		Map<String, Set<String>> tPos = titlesPosMap.get(title);
		Set<String> titlesHypSet = new HashSet<>();
		for (Map.Entry<String, Set<String>> h : tPos.entrySet()) {
			if (h.getKey().equals(HypernymSimilarityWithArguments.VERB)) {
				Set<String> verbs = h.getValue();
				titlesHypSet.addAll(getHypernymsForAll(verbs, HypernymSimilarityWithArguments.VERB));
			}

			if (h.getKey().equals(HypernymSimilarityWithArguments.NOUN)) {
				Set<String> nouns = h.getValue();
				titlesHypSet.addAll(getHypernymsForAll(nouns, HypernymSimilarityWithArguments.NOUN));
			}
		}
		double[] titleGoogleVector = getGoogleVectorRepresentation(titlesHypSet);
		Map<String, Set<String>> bPos = bodiesPosMap.get(bodyId);
		// System.out.println("bPos.size()= " + bPos.size());
		Set<String> bodiesHypSet = new HashSet<>();

		for (Map.Entry<String, Set<String>> h : bPos.entrySet()) {
			if (h.getKey().equals(HypernymSimilarityWithArguments.VERB)) {
				Set<String> verbs = h.getValue();
				bodiesHypSet.addAll(getHypernymsForAll(verbs, HypernymSimilarityWithArguments.VERB));
			}

			if (h.getKey().equals(HypernymSimilarityWithArguments.NOUN)) {
				Set<String> nouns = h.getValue();
				bodiesHypSet.addAll(getHypernymsForAll(nouns, HypernymSimilarityWithArguments.NOUN));
			}
		}
		double[] bGoogleVector = getGoogleVectorRepresentation(bodiesHypSet);

		INDArray tVec_ = Nd4j.create(titleGoogleVector);
		INDArray bVec_ = Nd4j.create(bGoogleVector);
		double sim = Transforms.cosineSim(tVec_, bVec_);

		// System.out.println("features.size()= " + features.size());
		return sim;
	}

	private static double[] getGoogleVectorRepresentation(Set<String> words) {
		if (googleVec == null)
			loadGoogleNewsVec();

		INDArray identity = Nd4j.zeros(1, 300);

		// System.out.println(identity.shapeInfoToString());

		ArrayList<INDArray> m = new ArrayList<>();
		for (String tok : words) {
			INDArray tokVec; //
			if (googleVec.hasWord(tok)) {
				// getting vector for each word method
				// System.out.println("hehe " +
				// Nd4j.create(vec.getWordVector(tok.trim())));
				if (Nd4j.create(googleVec.getWordVector(tok)) != null)
					tokVec = Nd4j.create(googleVec.getWordVector(tok));
				else
					continue;
				// System.out.println(tok);
			} else
				tokVec = identity;
			m.add(tokVec);
		}
		INDArray result = m.stream().reduce(identity, INDArray::add);
		double[] arrResult = getArrayVec(result);
		if (arrResult == null)
			System.out.println("null" + " " + words);
		return arrResult;
	}

	private Set<String> getHypernymsForAll(Set<String> words, String POSTag) throws IOException {
		if (dict == null)
			dict = getDictionary();
		Set<String> hyps = new HashSet<>();
		words = FeatureExtractorWithModifiedBL.removeStopwords(words);
		for (String w : words) {
			hyps.addAll(getHypernyms(w.toLowerCase(), POSTag));
		}

		return hyps;
	}

	private Set<String> getHypernyms(String word, String POSTag) {
		// get the synset
		POS pos = null;
		if (POSTag.equals(HypernymSimilarityWithArguments.VERB))
			pos = POS.VERB;
		else if (POSTag.equals(HypernymSimilarityWithArguments.NOUN))
			pos = POS.NOUN;
		else if (POSTag.equals(HypernymSimilarityWithArguments.ADJ))
			pos = POS.ADJECTIVE;

		Set<String> hypernym = new HashSet<>();

		IIndexWord idxWord = dict.getIndexWord(word, pos);
		if (idxWord != null) {
			IWordID wordID = idxWord.getWordIDs().get(0);
			IWord wordi = dict.getWord(wordID);
			ISynset synset = wordi.getSynset();

			// get the hypernyms
			List<ISynsetID> hypernyms = synset.getRelatedSynsets(Pointer.HYPERNYM);

			// print out each hypernyms id and synonyms

			List<IWord> words;

			for (ISynsetID sid : hypernyms) {
				words = dict.getSynset(sid).getWords();
				for (Iterator<IWord> i = words.iterator(); i.hasNext();) {
					hypernym.add(i.next().getLemma());
				}

			}
		}
		return hypernym;
	}

	private IDictionary getDictionary() throws IOException {
		// construct the URL to the Wordnet dictionary directory
		String path = ProjectPaths.WORDNET_DICT;
		URL url = null;
		try {
			url = new URL("file", null, path);
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}
		if (url == null)
			return null;

		// construct the dictionary object and open it
		IDictionary dict = new Dictionary(url);
		dict.open();
		return dict;
	}

	private double calcDistance(String title, String str2) {
		title = FeatureExtractorWithModifiedBL.getLemmatizedCleanStr(title);
		List<String> h = FeatureExtractorWithModifiedBL.removeStopWords(Arrays.asList(title.split("\\s+")));
		// get the string back
		StringBuilder sb = new StringBuilder();
		for (String s : h) {
			sb.append(s);
			sb.append(" ");
		}
		title = sb.toString().trim();

		str2 = FeatureExtractorWithModifiedBL.getLemmatizedCleanStr(str2);
		List<String> str2L = FeatureExtractorWithModifiedBL.removeStopWords(Arrays.asList(str2.split("\\s+")));
		// get the string back
		StringBuilder sb1 = new StringBuilder();
		for (String s : str2L) {
			sb1.append(s);
			sb1.append(" ");
		}
		str2 = sb1.toString().trim();

		// return cosSimMetric.distance(title, str2);
		return 0;
	}

	private static double[] getArrayVec(INDArray result) {
		double[] arrResult = result.data().asDouble();

		return arrResult;
	}

	private static StanfordCoreNLP getStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}

	public static FileHashMap<Integer, Map<String, Set<String>>> loadBodiesPosAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<Integer, Map<String, Set<String>>> posMap = new FileHashMap<Integer, Map<String, Set<String>>>(
				hashFileName, FileHashMap.FORCE_OVERWRITE);
		return posMap;
	}

	public static FileHashMap<String, Map<String, Set<String>>> loadtitlesPosAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Map<String, Set<String>>> posMap = new FileHashMap<String, Map<String, Set<String>>>(
				hashFileName, FileHashMap.FORCE_OVERWRITE);
		return posMap;
	}

	public static FileHashMap<String, Double> loadHypSimAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Double> hypSim = new FileHashMap<String, Double>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return hypSim;
	}

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		trainingIdBoyMap = sddr.getTrainIdBodyMap();
		testIdBoyMap = sddr.getTestIdBodyMap();

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	private static void loadGoogleNewsVec() {
		googleVec = WordVectorSerializer.readWord2VecModel("C:/thesis_stuff/GoogleNews-vectors-negative300.bin.gz",
				false);
	}

	public static void main(String[] args) throws Exception {
		loadData();
		// StanfordDependencyParser sdp = new StanfordDependencyParser();
		// System.out.println(sdp.getDependenciesAsTxtList("The sun shines every
		// day."));

		// System.out.println(hs.getPosMap(
		// "The sun shines every day. And how lovely to be with you. I am
		// smelling the flowers and my husband went to work."));
		/*
		 * Set<String> allTitles = new HashSet<>(); for (List<String> ts :
		 * trainingStances) allTitles.add(ts.get(0)); for (List<String> ts :
		 * testStances) allTitles.add(ts.get(0));
		 * 
		 * hs.saveTitlePOSTagMapFile(allTitles,
		 * ProjectPaths.TITLES_POS_MAP_PATH);
		 */
		HypernymSimilarityWithArguments hs = new HypernymSimilarityWithArguments();
		HashMap<Integer, String> allBodies = new HashMap<>();
		allBodies.putAll(testIdBoyMap);
		allBodies.putAll(trainingIdBoyMap);
		//hs.saveBodiesPOSTagMapFile(allBodies, ProjectPaths.BODIES_ARGS_POS_MAP_PATH);

		//FileHashMap<Integer, Map<String, Set<String>>> bb = loadBodiesPosAsHashFile(
		//		ProjectPaths.BODIES_ARGS_POS_MAP_PATH);
		//System.out.println(bb.get(476));

		hs.getHypSimFeatureVector(trainingStances, trainingIdBoyMap, ProjectPaths.TRAIN_ARG_HYP_SIM_PATH);
		hs.getHypSimFeatureVector(testStances, testIdBoyMap, ProjectPaths.TEST_ARG_HYP_SIM_PATH);
	}

}
