package ude.master.thesis.stance_detection.processor;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;
import org.tartarus.snowball.ext.englishStemmer;

import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class StanfordDependencyParser {

	private static boolean pipeInitialized = false;
	private static StanfordCoreNLP pipeline;

	public static Annotation buildAnnotatedDoc(String text) {

		if (!pipeInitialized)
			initPipe();

		// create an empty Annotation just with the given text
		Annotation document = new Annotation(text);

		// run all Annotators on this text
		pipeline.annotate(document);

		return document;

	}

	private static void initPipe() {
		// creates a StanfordCoreNLP object, with POS tagging, lemmatization,
		// NER, parsing, and coreference resolution
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		pipeline = new StanfordCoreNLP(props);
		pipeInitialized = true;
	}

	public static List<SemanticGraph> buildDependencyGraph(Annotation document) {
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		List<SemanticGraph> allSentencesDependencies = new ArrayList<SemanticGraph>();

		for (CoreMap sentence : sentences) {
			// this is the Stanford dependency graph of the current sentence
			SemanticGraph dependencies = sentence.get(EnhancedPlusPlusDependenciesAnnotation.class);
			allSentencesDependencies.add(dependencies);
		}
		return allSentencesDependencies;
	}

	public static List<String> lemmatize(Annotation document) {
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		List<String> lemmas = new ArrayList<>();
		for (CoreMap sentence : sentences) {
			// Iterate over all tokens in a sentence
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// Retrieve and add the lemma for each word into the
				// list of lemmas
				lemmas.add(token.get(LemmaAnnotation.class));
			}
		}
		return lemmas;
	}

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				"resources/data/summ_train_bodies.csv", "resources/data/test_data/summ_competition_test_bodies.csv");

		// HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
		// .readSummIdBodiesMap(new
		// File("resources/data/train_bodies_preprocessed_summ.csv"));
		// generaateRootDistFeaturesAndSave(trainingSummIdBoyMap,
		// "C:/thesis_stuff/features/train_rootdist.csv");

		// HashMap<Integer, Map<Integer, String>> testSummIdBoyMap =
		// sddr.readSummIdBodiesMap(new
		// File("resources/data/test_data/test_bodies_preprocessed_summ.csv"));
		// generaateRootDistFeaturesAndSave(testSummIdBoyMap,
		// "C:/thesis_stuff/features/test_rootdist.csv");

		/*
		 * String text3 =
		 * "Iraq Says Arrested Woman Is Not The Wife of ISIS Leader al-Baghdadi"
		 * ; Annotation doc = buildAnnotatedDoc(text3); List<SemanticGraph>
		 * graphs = buildDependencyGraph(doc); List<Integer> rootDistVec =
		 * getRooDistFeatureVec(graphs); System.out.println(rootDistVec);
		 */

		// Task: save dependencies already
		saveDepsListsMaps();

		String txt = "Kim Jong-un has broken both of his ankles and is now in the hospital after undergoing "
				+ "surgery, a report in a South Korean newspaper claims. The North Korean leader has "
				+ "been missing for more than three weeks, fueling speculation about what could cause his "
				+ "unusual disappearance from the public eye. This rumor seems to confirm what North Korean "
				+ "state media had said on Thursday, when state broadcaster Korean Central Television "
				+ "reported that Kim was not feeling well, and was suffering from an uncomfortable "
				+ "physical condition. Have something to add to this story? Share it in the comments.";
		// System.out.println(getDependenciesAsTxtList(txt));
		// System.out.println(FeatureExtractor.getLemmatizedCleanStr("I am
		// happier than ever."));
	}

	private static void generaateRootDistFeaturesAndSave(HashMap<Integer, Map<Integer, String>> summIdBodyMap,
			String filename) throws IOException {
		List<String[]> entries = new ArrayList<>();

		String[] header = new String[21];
		header[0] = "Body ID";
		for (int j = 1; j < 10; j++) {
			if (j >= 1 && j <= 6) {
				header[j] = "beg_disc_RootDist" + j;
			} else {
				header[j] = "end_disc_RootDist" + j;
			}
		}

		for (int j = 11; j < 21; j++) {
			if (j >= 11 && j <= 16) {
				header[j] = "beg_ref_RootDist" + j;
			} else {
				header[j] = "end_ref_RootDist" + j;
			}
		}
		entries.add(header);

		int i = 0;
		for (Map.Entry<Integer, Map<Integer, String>> e : summIdBodyMap.entrySet()) {
			List<String> entry = new ArrayList<>();
			entry.add(Integer.toString(e.getKey()));

			// add the discuss root dist
			for (int k = 1; k <= 3; k++) {
				// don't take the middle part
				if (k != 2) {
					String partText = e.getValue().get(k);
					Annotation doc = buildAnnotatedDoc(partText);
					// System.out.println(e.getValue());
					List<SemanticGraph> graphs = buildDependencyGraph(doc);

					// for Discuss rootDist
					List<Double> discussRootDistVec = getDiscussRootDistFeatureVec(graphs, k);

					for (Double d : discussRootDistVec)
						entry.add(Double.toString(d));

				}

			}

			// add the refute root dist
			for (int k = 1; k <= 3; k++) {
				// don't take the middle part
				if (k != 2) {
					String partText = e.getValue().get(k);
					Annotation doc = buildAnnotatedDoc(partText);
					// System.out.println(e.getValue());
					List<SemanticGraph> graphs = buildDependencyGraph(doc);

					// for Refute rootDist
					List<Double> refuteRootDistVec = getRefuteRootDistFeatureVec(graphs, k);

					for (Double d : refuteRootDistVec)
						entry.add(Double.toString(d));

				}

			}
			// System.out.println();
			// System.out.println(entry.size() + " : " + entry);
			entries.add(entry.toArray(new String[0]));
			i++;
			// if (i == 10)
			// break;
			if (i % 100 == 0)
				System.out.println("processed: " + i);
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(filename))) {
			writer.writeAll(entries);
		}

	}

	private static List<Double> getDiscussRootDistFeatureVec(List<SemanticGraph> graphs, int partLocation) {

		List<String> discuss_words = Arrays.asList(FeatureExtractor.discussWordsJoined);

		// double avgr = 0;
		// int ir = 0;

		// double avgd = 0;
		// int id = 0;

		List<Double> mins_dist = new ArrayList<>();
		// get the min rootDist for each graph (2 values vector for each)
		// System.out.println("graphs.size = " + graphs.size());
		for (SemanticGraph graph : graphs) {
			int minDiscussDist = 1000;

			IndexedWord root = graph.getFirstRoot(); // TODO: it it precise to
														// take only the first
														// root
			if (graph.getRoots().size() > 1)
				System.out.println("num of roots: " + graph.getRoots().size());

			// traverse the nodes in the graph
			for (int i = 1; i <= graph.size(); i++) {
				// System.out.println("graph size: " + graph.size());
				IndexedWord idxW = graph.getNodeByIndexSafe(i);
				// System.out.println("is idxW null " + idxW == null);
				// System.out.println(idxW.word());
				if (idxW != null)
					if (idxW.word() != null)
						if ((discuss_words.contains(idxW.word().toLowerCase()))) {
							// System.out.println("found discuss: " +
							// graph.getNodeByIndex(i).word());
							int dist = graph.getShortestDirectedPathEdges(root, graph.getNodeByIndex(i)).size();
							// System.out.println(graph.getShortestDirectedPathEdges(root,
							// graph.getNodeByIndex(i)));

							// avgd += dist;
							// id++;
							if (dist < minDiscussDist)
								minDiscussDist = dist;
						}
			}
			mins_dist.add((double) minDiscussDist);

		}

		int s = mins_dist.size();
		if (partLocation == 1) {
			if (s < FeatureExtractor.NO_SENTENCES_BEG) {
				while (mins_dist.size() < FeatureExtractor.NO_SENTENCES_BEG)
					mins_dist.add(1000.0);
			}
		} else {
			if (partLocation == 3)
				if (s < FeatureExtractor.NO_SENTENCES_END) {
					while (mins_dist.size() < FeatureExtractor.NO_SENTENCES_END)
						mins_dist.add(1000.0);
				}
		}
		// System.out.println("mins_dist.size " + mins_dist.size());
		return mins_dist;
	}

	private static List<Double> getRefuteRootDistFeatureVec(List<SemanticGraph> graphs, int partLocation) {

		List<String> refuting_words = Arrays.asList(FeatureExtractor.refutingWords);

		List<Double> mins_dist = new ArrayList<>();
		// get the min rootDist for each graph (2 values vector for each)
		// System.out.println("graphs.size = " + graphs.size());
		for (SemanticGraph graph : graphs) {
			int minRefuteDist = 1000;

			IndexedWord root = graph.getFirstRoot(); // TODO: it it precise to
														// take only the first
														// root
			if (graph.getRoots().size() > 1)
				System.out.println("num of roots: " + graph.getRoots().size());

			// traverse the nodes in the graph
			for (int i = 1; i <= graph.size(); i++) {
				// Systedm.out.println("graph size: " + graph.size());
				IndexedWord idxW = graph.getNodeByIndexSafe(i);

				if ((idxW != null))
					if (idxW.word() != null)
						if ((refuting_words.contains(idxW.word().toLowerCase()))) {
							// System.out.println("found refute: " +
							// graph.getNodeByIndex(i).word());
							int dist = graph.getShortestDirectedPathEdges(root, graph.getNodeByIndex(i)).size();

							if (dist < minRefuteDist)
								minRefuteDist = dist;
						}

			}
			mins_dist.add((double) minRefuteDist);

		}

		int s = mins_dist.size();
		if (partLocation == 1) {
			if (s < FeatureExtractor.NO_SENTENCES_BEG) {
				while (mins_dist.size() < FeatureExtractor.NO_SENTENCES_BEG)
					mins_dist.add(1000.0);
			}
		} else {
			if (partLocation == 3)
				if (s < FeatureExtractor.NO_SENTENCES_END) {
					while (mins_dist.size() < FeatureExtractor.NO_SENTENCES_END)
						mins_dist.add(1000.0);
				}
		}

		// System.out.println("mins_dist.size " + mins_dist.size());
		return mins_dist;
	}

	/**
	 * For bodies: Saves the dependencies as a file map with(body_id ,map(key: the part No.
	 * , value: a list of dependencies for each sentence in the part))
	 * 
	 * For titles: Saves the dependencies as a file map with (key: the title
	 * text, value: list of dependencies for title's sentences
	 * 
	 * @throws IOException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 */
	public static void saveDepsListsMaps()
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				"resources/data/train_stances_preprocessed.csv", "resources/data/train_bodies_preprocessed_summ.csv",
				"resources/data/test_data/test_stances_preprocessed.csv",
				"resources/data/test_data/test_bodies_preprocessed_summ.csv");

		// ******************* Do training data ****************************
		// Map<Integer, String> trainIdBodyMap = sddr.getTrainIdBodyMap();
		// Use summarized bodies instead
		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File("resources/data/train_bodies_preprocessed_summ.csv"));
		List<List<String>> trainingStances = sddr.getTrainStances();

		// Do the training bodies
		FileHashMap<Integer, Map<Integer, List<String>>> trainBDepsMap = new FileHashMap<Integer, Map<Integer, List<String>>>(
				"C:/thesis_stuff/help_files/train_bodies_deps", FileHashMap.FORCE_OVERWRITE);

		for (Entry<Integer, Map<Integer, String>> b : trainingSummIdBoyMap.entrySet()) {
			Map<Integer, List<String>> partsDeps = new HashMap<>();

			for (int i = 1; i <= 3; i++) {
				List<String> deps = new ArrayList<>();
				if (!b.getValue().get(i).isEmpty()) {
					deps = getDependenciesAsTxtList(b.getValue().get(i));
				}
				partsDeps.put(i, deps);
			}

			trainBDepsMap.put(b.getKey(), partsDeps);
		}
		trainBDepsMap.save();
		trainBDepsMap.close();

		// Do the titles
		FileHashMap<String, List<String>> trainTDepsMap = new FileHashMap<String, List<String>>(
				"C:/thesis_stuff/help_files/train_titles_deps", FileHashMap.FORCE_OVERWRITE);
		Set<String> testTitlesSet = new HashSet<>();
		for (List<String> s : trainingStances) {
			testTitlesSet.add(s.get(0));
		}
		for (String t : testTitlesSet) {
			List<String> deps = getDependenciesAsTxtList(t);
			trainTDepsMap.put(t, deps);
		}
		trainTDepsMap.save();
		trainTDepsMap.close();

		// ******************* Do Test data ****************************
		// HashMap<Integer, String> testIdBodyMap = sddr.getTestIdBodyMap();
		// Use summarized bodies instead
		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File("resources/data/test_data/test_bodies_preprocessed_summ.csv"));
		List<List<String>> testStances = sddr.getTestStances();

		// Do the training bodies
		FileHashMap<Integer, Map<Integer, List<String>>> testBDepsMap = new FileHashMap<Integer, Map<Integer, List<String>>>(
				"C:/thesis_stuff/help_files/test_bodies_deps", FileHashMap.FORCE_OVERWRITE);
		for (Entry<Integer, Map<Integer, String>> b : testSummIdBoyMap.entrySet()) {
			Map<Integer, List<String>> partsDeps = new HashMap<>();

			for (int i = 1; i <= 3; i++) {
				List<String> deps = new ArrayList<>();
				if (!b.getValue().get(i).isEmpty()) {
					deps = getDependenciesAsTxtList(b.getValue().get(i));
				}
				partsDeps.put(i, deps);
			}

			testBDepsMap.put(b.getKey(), partsDeps);
		}
		testBDepsMap.save();
		testBDepsMap.close();

		// Do the titles
		FileHashMap<String, List<String>> testTDepsMap = new FileHashMap<String, List<String>>(
				"C:/thesis_stuff/help_files/test_titles_deps", FileHashMap.FORCE_OVERWRITE);
		Set<String> trainTitlesSet = new HashSet<>();
		for (List<String> s : testStances) {
			trainTitlesSet.add(s.get(0));
		}
		for (String t : trainTitlesSet) {
			List<String> deps = getDependenciesAsTxtList(t);
			testTDepsMap.put(t, deps);
		}
		testTDepsMap.save();
		testTDepsMap.close();
	}

	/**
	 * 
	 * @param text
	 *            the part of the body with a distinct number of sentences
	 * @return a list of dependencies; a dependency for each sentence in the
	 *         text
	 */
	private static List<String> getDependenciesAsTxtList(String text) {
		Annotation doc = StanfordDependencyParser.buildAnnotatedDoc(text);

		List<SemanticGraph> graphs = StanfordDependencyParser.buildDependencyGraph(doc);

		List<String> depLists = new ArrayList<>();
		// Sentence / Graph Index
		for (SemanticGraph graph : graphs) {

			String depList = graph.toList();

			int size = graph.size();
			// printing stuff to see
			/*
			 * for(int j = 1; j < size; j++){ String l =
			 * graph.getNodeByIndexSafe(j).lemma(); String tag =
			 * graph.getNodeByIndexSafe(j).tag(); System.out.println(l + " " +
			 * tag); } System.out.println(graph.toRecoveredSentenceString());
			 */
			depLists.add(depList);
		}
		return depLists;
	}

}
