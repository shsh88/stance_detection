package ude.master.thesis.stance_detection.processor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class StanfordDependencyParser {

	public static Annotation buildAnnotatedDoc(String text) {

		// creates a StanfordCoreNLP object, with POS tagging, lemmatization,
		// NER, parsing, and coreference resolution
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

		// create an empty Annotation just with the given text
		Annotation document = new Annotation(text);

		// run all Annotators on this text
		pipeline.annotate(document);

		return document;

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

	public static void main(String[] args) throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				"resources/data/summ_train_bodies.csv", "resources/data/test_data/summ_competition_test_bodies.csv");

		// Map<Integer, String> trainIdBodyMap = sddr.getTrainIdBodyMap();
		// generaateRootDistFeaturesAndSave(trainIdBodyMap,
		// "C:/thesis_stuff/features/train_rootdist.csv");
		HashMap<Integer, String> testIdBodyMap = sddr.getTestIdBodyMap();
		generaateRootDistFeaturesAndSave(testIdBodyMap, "C:/thesis_stuff/features/test_rootdist.csv");

		/*
		 * String text3 =
		 * "Iraq Says Arrested Woman Is Not The Wife of ISIS Leader al-Baghdadi"
		 * ; Annotation doc = buildAnnotatedDoc(text3); List<SemanticGraph>
		 * graphs = buildDependencyGraph(doc); List<Integer> rootDistVec =
		 * getRooDistFeatureVec(graphs); System.out.println(rootDistVec);
		 */
	}

	private static void generaateRootDistFeaturesAndSave(Map<Integer, String> trainIdBodyMap, String filename)
			throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "Body ID", "ref_RootDist", "disc_RootDist", "ref_avg", "disc_avg" });
		int i = 0;
		for (Map.Entry<Integer, String> e : trainIdBodyMap.entrySet()) {
			Annotation doc = buildAnnotatedDoc(e.getValue());
			// System.out.println(e.getValue());
			List<SemanticGraph> graphs = buildDependencyGraph(doc);
			List<Double> rootDistVec = getRooDistFeatureVec(graphs);

			List<String> entry = new ArrayList<>();
			entry.add(Integer.toString(e.getKey()));
			for (Double d : rootDistVec)
				entry.add(Double.toString(d));

			entries.add(entry.toArray(new String[0]));

			i++;
			// if (i == 10)
			// break;
			if (i % 50 == 0)
				System.out.println("processed: " + i);
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(filename))) {
			writer.writeAll(entries);
		}
	}

	private static List<Double> getRooDistFeatureVec(List<SemanticGraph> graphs) {

		List<String> refuting_words = Arrays.asList(FeatureExtractor.refutingWords);
		List<String> discuss_words = Arrays.asList(FeatureExtractor.discussWords);

		// List<String> lemmas = lemmatize(doc);

		int minRefuteDist = 1000;
		int minDiscussDist = 1000;

		double avgr = 0;
		int ir = 0;

		double avgd = 0;
		int id = 0;

		for (SemanticGraph graph : graphs) {
			IndexedWord root = graph.getFirstRoot();
			for (int i = 1; i <= graph.size(); i++) {
				// System.out.println("graph size: " + graph.size());
				IndexedWord idxW = graph.getNodeByIndexSafe(i);
				System.out.println("is idxW null " + idxW == null);
				//System.out.println(idxW.word());

				if ((idxW != null))
					if (idxW.word() != null)
						if ((refuting_words.contains(idxW.word().toLowerCase()))) {
							// System.out.println("found refute: " +
							// graph.getNodeByIndex(i).word());
							int dist = graph.getShortestDirectedPathEdges(root, graph.getNodeByIndex(i)).size();
							System.out.println(graph.getShortestDirectedPathEdges(root, graph.getNodeByIndex(i)));

							avgr += dist;
							ir++;
							if (dist < minRefuteDist)
								minRefuteDist = dist;
						}
				if (idxW != null)
					if (idxW.word() != null)
						if ((discuss_words.contains(idxW.word().toLowerCase()))) {
							// System.out.println("found discuss: " +
							// graph.getNodeByIndex(i).word());
							int dist = graph.getShortestDirectedPathEdges(root, graph.getNodeByIndex(i)).size();
							// System.out.println(graph.getShortestDirectedPathEdges(root,
							// graph.getNodeByIndex(i)));

							avgd += dist;
							id++;
							if (dist < minDiscussDist)
								minDiscussDist = dist;
						}
			}

		}

		List<Double> mins_avgs = new ArrayList<>();
		mins_avgs.add((double) minRefuteDist);
		mins_avgs.add((double) minDiscussDist);

		if (ir == 0)
			mins_avgs.add((double) 1000);
		else
			mins_avgs.add(avgr / ir);

		if (id == 0)
			mins_avgs.add((double) 1000);
		else
			mins_avgs.add(avgd / id);

		return mins_avgs;
	}

}
