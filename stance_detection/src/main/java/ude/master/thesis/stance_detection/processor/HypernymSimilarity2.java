package ude.master.thesis.stance_detection.processor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
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

import com.opencsv.CSVReader;
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
import lombok.val;
import ude.master.thesis.stance_detection.util.BodySummerizer2;
import ude.master.thesis.stance_detection.util.PPDBProcessor;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class HypernymSimilarity2 {
	protected StanfordCoreNLP pipeline;
	public static final String VERB = "verb";
	public static final String NOUN = "noun";
	public static final String ADJ = "Adjective";

	private static Word2Vec googleVec;

	private static FileHashMap<Integer, Map<Integer, ArrayList<Map<String, Set<String>>>>> bodiesPosMap;
	private static FileHashMap<String, Map<String, Set<String>>> titlesPosMap;

	private Map<Integer, String> trainIdBodyMap = new HashMap<Integer, String>();
	private static List<List<String>> trainingStances = new ArrayList<>();
	private HashMap<Integer, String> testIdBodyMap = new HashMap<>();
	private IDictionary dict;
	private static List<List<String>> testStances = new ArrayList<List<String>>();
	private static HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = new HashMap<>();
	private static HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = new HashMap<>();

	public HypernymSimilarity2() {
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

	public void saveBodiesPOSTagMapFile(HashMap<Integer, Map<Integer, String>> summIdBoyMap, String filepath)
			throws Exception {

		// HashMap to save values
		FileHashMap<Integer, Map<Integer, ArrayList<Map<String, Set<String>>>>> posMap = new FileHashMap<Integer, Map<Integer, ArrayList<Map<String, Set<String>>>>>(
				filepath, FileHashMap.FORCE_OVERWRITE);

		for (Entry<Integer, Map<Integer, String>> bodyParts : summIdBoyMap.entrySet()) {
			System.out.println(bodyParts.getValue().size());
			Map<Integer, ArrayList<Map<String, Set<String>>>> featureVec = new HashMap<>();
			for (int i = 1; i <= 3; i++) {
				if ((i == 1) || (i == 3)) {
					String part = bodyParts.getValue().get(i);
					ArrayList<Map<String, Set<String>>> subVec = getPOSMapPart(part, i);
					featureVec.put(i, subVec);

				} else {
					Map<String, Set<String>> fAll = getPOSMapAll(bodyParts.getValue().get(i));
					ArrayList<Map<String, Set<String>>> subVec = new ArrayList<>();
					subVec.add(fAll);
					featureVec.put(i, subVec);

				}
			}

			posMap.put(bodyParts.getKey(), featureVec);

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

	private ArrayList<Map<String, Set<String>>> getPOSMapPart(String part, int partNo) {
		ArrayList<Map<String, Set<String>>> subVec = new ArrayList<>();

		if ((partNo == 1) || (partNo == 3)) {

			Annotation doc = new Annotation(part);
			pipeline.annotate(doc);
			List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

			for (CoreMap s : sentences) {
				Map<String, Set<String>> f = getPosMap(s.toString());
				subVec.add(f);
			}
		}

		return subVec;
	}

	public void getHypSimFeatureVector(List<List<String>> stances, HashMap<Integer, Map<Integer, String>> summIdBoyMap,
			String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, ArrayList<Double>> hypSims = new FileHashMap<String, ArrayList<Double>>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		if ((bodiesPosMap == null) && (titlesPosMap == null)) {
			bodiesPosMap = loadBodiesPosAsHashFile(ProjectPaths.BODIES_POS_MAP_PATH2);
			titlesPosMap = loadtitlesPosAsHashFile(ProjectPaths.TITLES_POS_MAP_PATH);
		}

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		for (int i = 0; i < 9; i++) {
			csvHeader.add("hyp_sim_" + i);
		}

		entries.add(csvHeader.toArray(new String[0]));

		int k = 0;
		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			int bodyId = Integer.valueOf(s.get(1));

			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			Map<Integer, String> bodyParts = summIdBoyMap.get(Integer.valueOf(s.get(1)));

			ArrayList<Double> featureVec = new ArrayList<>();
			for (int i = 1; i <= 3; i++) {
				String part = bodyParts.get(i);
				List<Double> subVec = getHypSimfeaturePart(title, part, bodyId, i);
				featureVec.addAll(subVec);

				for (Double v : subVec)
					entry.add(v.toString());

			}
			// System.out.println(entry.size());
			
			System.out.println(entries.size());
			if (entry.size() != 12)
				throw new Exception("not 9 features");

			hypSims.put(title + s.get(1), featureVec);

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

	private Double getCosSimfeatureAll(String title, String body) {
		return calcDistance(title, body);
	}

	private List<Double> getHypSimfeaturePart(String title, String part, int bodyId, int partNo) throws IOException {
		List<Double> subVec = new ArrayList<>();

		if (part.equals(""))
			if (partNo == 1) {
				Double[] vec = new Double[BodySummerizer2.NUM_SENT_BEG];
				Arrays.fill(vec, -100.0);
				return Arrays.asList(vec);
			} else if (partNo == 3) {
				Double[] vec = new Double[BodySummerizer2.NUM_SENT_END];
				Arrays.fill(vec, -100.0);
				return Arrays.asList(vec);
			}

		List<Double> f = calcHypSim(title, bodyId, partNo);
		subVec.addAll(f);
		if (partNo == 1)
			while (subVec.size() < BodySummerizer2.NUM_SENT_BEG)
				subVec.add(-100.0);

		if (partNo == 2)
			while (subVec.size() < 1)
				subVec.add(-100.0);

		if (partNo == 3)
			while (subVec.size() < BodySummerizer2.NUM_SENT_END)
				subVec.add(-100.0);

		return subVec;
	}

	private List<Double> calcHypSim(String title, int bodyId, int partNo) throws IOException {
		List<Double> features = new ArrayList<>();

		Map<String, Set<String>> tPos = titlesPosMap.get(title);
		Set<String> titlesHypSet = new HashSet<>();
		for (Map.Entry<String, Set<String>> h : tPos.entrySet()) {
			if (h.getKey().equals(HypernymSimilarity2.VERB)) {
				Set<String> verbs = h.getValue();
				titlesHypSet.addAll(getHypernymsForAll(verbs, HypernymSimilarity2.VERB));
			}

			if (h.getKey().equals(HypernymSimilarity2.NOUN)) {
				Set<String> nouns = h.getValue();
				titlesHypSet.addAll(getHypernymsForAll(nouns, HypernymSimilarity2.NOUN));
			}
		}
		double[] titleGoogleVector = getGoogleVectorRepresentation(titlesHypSet);
		ArrayList<Map<String, Set<String>>> bPos = bodiesPosMap.get(bodyId).get(partNo);
		// System.out.println("bPos.size()= " + bPos.size());
		for (Map<String, Set<String>> pp : bPos) {
			Set<String> bodiesHypSet = new HashSet<>();

			for (Map.Entry<String, Set<String>> h : pp.entrySet()) {
				if (h.getKey().equals(HypernymSimilarity2.VERB)) {
					Set<String> verbs = h.getValue();
					bodiesHypSet.addAll(getHypernymsForAll(verbs, HypernymSimilarity2.VERB));
				}

				if (h.getKey().equals(HypernymSimilarity2.NOUN)) {
					Set<String> nouns = h.getValue();
					bodiesHypSet.addAll(getHypernymsForAll(nouns, HypernymSimilarity2.NOUN));
				}
			}
			double[] bGoogleVector = getGoogleVectorRepresentation(bodiesHypSet);

			INDArray tVec_ = Nd4j.create(titleGoogleVector);
			INDArray bVec_ = Nd4j.create(bGoogleVector);
			double sim = Transforms.cosineSim(tVec_, bVec_);
			features.add(sim);

		}
		// System.out.println("features.size()= " + features.size());
		return features;
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
		if (POSTag.equals(HypernymSimilarity2.VERB))
			pos = POS.VERB;
		else if (POSTag.equals(HypernymSimilarity2.NOUN))
			pos = POS.NOUN;
		else if (POSTag.equals(HypernymSimilarity2.ADJ))
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

	public static FileHashMap<Integer, Map<Integer, ArrayList<Map<String, Set<String>>>>> loadBodiesPosAsHashFile(
			String hashFileName) throws FileNotFoundException, ObjectExistsException, ClassNotFoundException,
			VersionMismatchException, IOException {
		FileHashMap<Integer, Map<Integer, ArrayList<Map<String, Set<String>>>>> posMap = new FileHashMap<Integer, Map<Integer, ArrayList<Map<String, Set<String>>>>>(
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

	public static FileHashMap<String, ArrayList<Double>> loadHypSimAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> hypSim = new FileHashMap<String, ArrayList<Double>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return hypSim;
	}

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES2,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES2);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES2));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES2));

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
		HypernymSimilarity2 hs = new HypernymSimilarity2();
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
		/*HashMap<Integer, Map<Integer, String>> allBodies = new HashMap<>();
		allBodies.putAll(testSummIdBoyMap);
		allBodies.putAll(trainingSummIdBoyMap);
		hs.saveBodiesPOSTagMapFile(allBodies, ProjectPaths.BODIES_POS_MAP_PATH2);

		FileHashMap<Integer, Map<Integer, ArrayList<Map<String, Set<String>>>>> bb = loadBodiesPosAsHashFile(
				ProjectPaths.BODIES_POS_MAP_PATH2);
		System.out.println(bb.get(476));*/

		//hs.getHypSimFeatureVector(trainingStances, trainingSummIdBoyMap, ProjectPaths.TRAIN_HYP_SIM_PATH2);
		//hs.getHypSimFeatureVector(testStances, testSummIdBoyMap, ProjectPaths.TEST_HYP_SIM_PATH2);
		//hs.saveHypSimFeatureVectorInHashFile(ProjectPaths.TEST_HYP_SIM_PATH2, ProjectPaths.TEST_HYP_SIM_PATH2+".csv");
		hs.saveHypSimFeatureVectorInHashFile(ProjectPaths.TRAIN_HYP_SIM_PATH2, ProjectPaths.TRAIN_HYP_SIM_PATH2+".csv");
	
	}

	private void saveHypSimFeatureVectorInHashFile(String hashFilePath, String csvFilePath) throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		FileHashMap<String, ArrayList<Double>> hashFile = new FileHashMap<String, ArrayList<Double>>(hashFilePath,
				FileHashMap.FORCE_OVERWRITE);

		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvFilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {
			ArrayList<Double> values = new ArrayList<>();
			
			for(int i = 3; i < line.length; i++)
				values.add(Double.valueOf(line[i]));
			
			hashFile.put(line[0] + line[1], values);
		}
		reader.close();

		// saving the map file
		hashFile.save();
		hashFile.close();
	}

}
