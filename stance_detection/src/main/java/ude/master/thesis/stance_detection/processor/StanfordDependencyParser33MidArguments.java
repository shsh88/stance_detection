package ude.master.thesis.stance_detection.processor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
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

import com.opencsv.CSVReader;
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
import ude.master.thesis.stance_detection.util.BodySummarizerWithArguments;
import ude.master.thesis.stance_detection.util.BodySummerizer2;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

/**
 * This class is the same as StanfordDependencyParser class but the body text
 * parts are changed
 * 
 * @author Razan
 *
 */
public class StanfordDependencyParser33MidArguments {

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

	public static void main(String[] args) throws Exception {

		// Task: save dependencies already
		// saveDepsListsMaps();

		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TEST);

		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN));
		generateRootDistFeaturesAndSave(trainingSummIdBoyMap, ProjectPaths.CSV_ROOT_DIST_FEATURE_PARTS_ARG33_TRAIN);

		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TEST));
		generateRootDistFeaturesAndSave(testSummIdBoyMap, ProjectPaths.CSV_ROOT_DIST_FEATURE_PARTS_ARG33_TEST);
		saveRootDistFeaturesAsHashFile(ProjectPaths.CSV_ROOT_DIST_FEATURE_PARTS_ARG33_TRAIN,
				ProjectPaths.ROOT_DIST_FEATURE_PARTS_ARG33_TRAIN);
		saveRootDistFeaturesAsHashFile(ProjectPaths.CSV_ROOT_DIST_FEATURE_PARTS_ARG33_TEST,
				ProjectPaths.ROOT_DIST_FEATURE_PARTS_ARG33_TEST);

	}

	/**
	 * 
	 * @param summIdBodyMap
	 * @param filename
	 * @throws Exception
	 */
	private static void generateRootDistFeaturesAndSave(HashMap<Integer, Map<Integer, String>> summIdBodyMap,
			String filename) throws Exception {
		List<String[]> entries = new ArrayList<>();

		String[] header = new String[23];
		header[0] = "Body ID";
		for (int j = 1; j < 7; j++) {
			if (j >= 1 && j <= 3) {
				header[j] = "beg_disc_RootDist" + j;
			} else {
				header[j] = "end_disc_RootDist" + j;
			}
		}

		for (int j = 7; j < 13; j++) {
			if (j >= 9 && j <= 11) {
				header[j] = "beg_ref_RootDist" + j;
			} else {
				header[j] = "end_ref_RootDist" + j;
			}
		}

		for (int j = 13; j < 19; j++) {
			if (j >= 13 && j <= 15) {
				header[j] = "mid_ref_RootDist" + j;
			} else {
				header[j] = "mid_disc_RootDist" + j;
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

			// add features from mid part
			// ******
			String partText = e.getValue().get(3);
			Annotation doc = buildAnnotatedDoc(partText);
			// System.out.println(e.getValue());
			List<SemanticGraph> graphs = buildDependencyGraph(doc);

			// for Refute rootDist
			List<Double> refuteRootDistVec = getRefuteRootDistFeatureVecFromMid(graphs);

			for (Double d : refuteRootDistVec)
				entry.add(Double.toString(d));

			List<Double> discussRootDistVec = getDiscussRootDistFeatureVecFromMid(graphs);
			for (Double d : discussRootDistVec)
				entry.add(Double.toString(d));

			// double avgDisc = getAvgRootDist(discussRootDistVec);
			// entry.add(Double.toString(avgDisc));

			// double avgRef = getAvgRootDist(refuteRootDistVec);
			// entry.add(Double.toString(avgRef));
			// ******

			// System.out.println(entry.size());
			if (entry.size() != 19)
				throw new Exception("not 18 features");
			entries.add(entry.toArray(new String[0]));
			i++;

			if (i % 1000 == 0)
				System.out.println("processed: " + i);
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(filename))) {
			writer.writeAll(entries);
		}

	}

	private static List<Double> getDiscussRootDistFeatureVecFromMid(List<SemanticGraph> graphs) {
		List<String> discuss_words = Arrays.asList(FeatureExtractorWithModifiedBL.discussWordsJoined);

		List<Double> mins_dist = new ArrayList<>();

		for (SemanticGraph graph : graphs) {
			int minDiscussDist = 1000;

			IndexedWord root = graph.getFirstRoot(); // TODO: it it precise to
														// take only the first
														// root
			if (graph.getRoots().size() > 1)
				System.out.println("num of roots: " + graph.getRoots().size());

			// traverse the nodes in the graph
			for (int i = 1; i <= graph.size(); i++) {
				IndexedWord idxW = graph.getNodeByIndexSafe(i);
				if (idxW != null)
					if (idxW.word() != null)
						if ((discuss_words.contains(idxW.word().toLowerCase()))) {
							int dist = graph.getShortestDirectedPathEdges(root, graph.getNodeByIndex(i)).size();

							if (dist < minDiscussDist)
								minDiscussDist = dist;
						}
			}
			mins_dist.add((double) minDiscussDist);

		}

		// get the values in min_dist that are smaller than 1000 first
		List<Double> min_dist_red = new ArrayList<>();
		for (Double m : mins_dist) {
			if (m < 1000.0)
				min_dist_red.add(m);
		}
		if (min_dist_red.size() >= 3)
			System.out.println("we got it >= 3!");

		while (min_dist_red.size() > 3) {
			min_dist_red.remove(min_dist_red.size() - 1);
		}
		while (min_dist_red.size() < 3)
			min_dist_red.add(1000.0);

		return min_dist_red;
	}

	private static List<Double> getRefuteRootDistFeatureVecFromMid(List<SemanticGraph> graphs) {
		List<String> refuting_words = Arrays.asList(FeatureExtractorWithModifiedBL.refutingWords);

		List<Double> mins_dist = new ArrayList<>();
		// get the min rootDist for each graph (2 values vector for each)
		// System.out.println("graphs.size = " + graphs.size());
		for (SemanticGraph graph : graphs) {
			int minRefuteDist = 1000;

			IndexedWord root = graph.getFirstRoot();

			if (graph.getRoots().size() > 1)
				System.out.println("num of roots: " + graph.getRoots().size());

			// traverse the nodes in the graph
			for (int i = 1; i <= graph.size(); i++) {
				IndexedWord idxW = graph.getNodeByIndexSafe(i);

				if ((idxW != null))
					if (idxW.word() != null)
						if ((refuting_words.contains(idxW.word().toLowerCase()))) {
							int dist = graph.getShortestDirectedPathEdges(root, graph.getNodeByIndex(i)).size();

							if (dist < minRefuteDist)
								minRefuteDist = dist;
						}

			}
			mins_dist.add((double) minRefuteDist);

		}

		// get the values in min_dist that are smaller than 1000 first
		List<Double> min_dist_red = new ArrayList<>();
		for (Double m : mins_dist) {
			if (m < 1000.0)
				min_dist_red.add(m);
		}
		if (min_dist_red.size() >= 3)
			System.out.println("we got it >= 3!");

		while (min_dist_red.size() > 3) {
			min_dist_red.remove(min_dist_red.size() - 1);
		}

		while (min_dist_red.size() < 3)
			min_dist_red.add(1000.0);

		return min_dist_red;
	}

	/**
	 * This represent 2 features: the distance from the root of a sentence to a
	 * refuting/discussing word saving the map as [
	 * <body_id>,["ref_RootDist","disc_RootDist"]] These 2 features are
	 * calculated for each sentence in the body (5 from the beginning and 3 at
	 * last) So the feature vector is of length 10
	 * 
	 * @param csvfilePath
	 * @param hashFileName
	 * @throws FileNotFoundException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 * @throws IOException
	 */
	public static void saveRootDistFeaturesAsHashFile(String csvfilePath, String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> rootDistData = new FileHashMap<String, ArrayList<Double>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {

			ArrayList<Double> values = new ArrayList<>();
			for (int i = 1; i < line.length; i++)
				values.add(Double.valueOf(line[i]));

			rootDistData.put(line[0], values);
		}
		reader.close();

		// saving the map file
		rootDistData.save();
		rootDistData.close();

	}

	public static FileHashMap<String, ArrayList<Double>> loadRootDistFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> rootDistData = new FileHashMap<String, ArrayList<Double>>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return rootDistData;

	}

	private static List<Double> getDiscussRootDistFeatureVec(List<SemanticGraph> graphs, int partLocation) {

		List<String> discuss_words = Arrays.asList(FeatureExtractorWithModifiedBL.discussWordsJoined);

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
			if (s < BodySummarizerWithArguments.NUM_SENT_BEG) {
				while (mins_dist.size() < BodySummarizerWithArguments.NUM_SENT_BEG)
					mins_dist.add(1000.0);
			}
		} else {
			if (partLocation == 3)
				if (s < BodySummarizerWithArguments.NUM_SENT_END) {
					while (mins_dist.size() < BodySummarizerWithArguments.NUM_SENT_END)
						mins_dist.add(1000.0);
				}
		}
		// System.out.println("mins_dist.size " + mins_dist.size());
		return mins_dist;
	}

	private static List<Double> getRefuteRootDistFeatureVec(List<SemanticGraph> graphs, int partLocation) {

		List<String> refuting_words = Arrays.asList(FeatureExtractorWithModifiedBL.refutingWords);

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
			if (s < BodySummarizerWithArguments.NUM_SENT_BEG) {
				while (mins_dist.size() < BodySummarizerWithArguments.NUM_SENT_BEG)
					mins_dist.add(1000.0);
			}
		} else {
			if (partLocation == 3)
				if (s < BodySummarizerWithArguments.NUM_SENT_END) {
					while (mins_dist.size() < BodySummarizerWithArguments.NUM_SENT_END)
						mins_dist.add(1000.0);
				}
		}

		// System.out.println("mins_dist.size " + mins_dist.size());
		return mins_dist;
	}

	/**
	 * For bodies: Saves the dependencies as a file map with(body_id ,map(key:
	 * the part No. , value: a list of dependencies for each sentence in the
	 * part))
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

		// StanceDetectionDataReader sddr = new StanceDetectionDataReader(true,
		// true, ProjectPaths.TRAIN_STANCES,
		// ProjectPaths.SUMMARIZED_TRAIN_BODIES, ProjectPaths.TEST_STANCESS,
		// ProjectPaths.SUMMARIZED_TEST_BODIES);
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TEST);

		// ******************* Do training data ****************************
		// Use summarized bodies instead
		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN));

		// Do the training bodies
		FileHashMap<Integer, Map<Integer, List<String>>> trainBDepsMap = new FileHashMap<Integer, Map<Integer, List<String>>>(
				ProjectPaths.TRAIN_BODIES_PARTS_ARG33_DEPS, FileHashMap.FORCE_OVERWRITE);

		for (Entry<Integer, Map<Integer, String>> b : trainingSummIdBoyMap.entrySet()) {
			Map<Integer, List<String>> partsDeps = new HashMap<>();

			for (int i = 1; i <= 3; i++) {
				List<String> deps = new ArrayList<>();
				deps = getDependenciesAsTxtList(b.getValue().get(i));
				partsDeps.put(i, deps);
			}

			trainBDepsMap.put(b.getKey(), partsDeps);
		}
		trainBDepsMap.save();
		trainBDepsMap.close();

		// Do the titles (we do not have to do this)

		// ******************* Do Test data ****************************
		// HashMap<Integer, String> testIdBodyMap = sddr.getTestIdBodyMap();
		// Use summarized bodies instead
		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TEST));

		// Do the training bodies
		FileHashMap<Integer, Map<Integer, List<String>>> testBDepsMap = new FileHashMap<Integer, Map<Integer, List<String>>>(
				ProjectPaths.TEST_BODIES_PARTS_ARG33_DEPS, FileHashMap.FORCE_OVERWRITE);
		for (Entry<Integer, Map<Integer, String>> b : testSummIdBoyMap.entrySet()) {
			Map<Integer, List<String>> partsDeps = new HashMap<>();

			for (int i = 1; i <= 3; i++) {
				List<String> deps = new ArrayList<>();
				deps = getDependenciesAsTxtList(b.getValue().get(i));
				partsDeps.put(i, deps);
			}

			testBDepsMap.put(b.getKey(), partsDeps);
		}
		testBDepsMap.save();
		testBDepsMap.close();
	}

	/**
	 * 
	 * @param text
	 *            the part of the body with a distinct number of sentences
	 * @return a list of dependencies; a dependency for each sentence in the
	 *         text/part
	 */
	public static List<String> getDependenciesAsTxtList(String text) {
		Annotation doc = StanfordDependencyParser33MidArguments.buildAnnotatedDoc(text);

		List<SemanticGraph> graphs = StanfordDependencyParser33MidArguments.buildDependencyGraph(doc);

		List<String> depLists = new ArrayList<>();
		// Sentence / Graph Index
		for (SemanticGraph graph : graphs) {

			String depList = graph.toList();

			depLists.add(depList);
		}
		return depLists;
	}

}
