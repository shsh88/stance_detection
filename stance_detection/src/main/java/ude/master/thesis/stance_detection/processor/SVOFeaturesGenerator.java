package ude.master.thesis.stance_detection.processor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.semgrex.SemgrexMatcher;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class SVOFeaturesGenerator {

	public static void main(String[] args) throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true, "resources/data/train_stances.csv",
				"resources/data/summ_train_bodies.csv", "resources/data/test_data/competition_test_stances.csv",
				"resources/data/test_data/summ_competition_test_bodies.csv");

		Map<Integer, String> trainIdBodyMap = sddr.getTrainIdBodyMap();
		List<List<String>> trainingStances = sddr.getTrainStances();
		HashMap<Integer, String> testIdBodyMap = sddr.getTestIdBodyMap();
		List<List<String>> testStances = sddr.getTestStances();

		String headline = "ISIL Beheads American Photojournalist in Iraq";
		String body = "James Foley, an American journalist who went missing in Syria more than a year ago, "
				+ "has reportedly been executed by the Islamic State, a militant group formerly known as ISIS. "
				+ "Video and photos purportedly of Foley emerged on Tuesday. A YouTube video -- entitled \"\"A "
				+ "Message to #America (from the #IslamicState)\"\" -- identified a man on his knees as"
				+ " \"\"James Wright Foley,\"\" and showed his execution. This is a developing story. "
				+ "Check back here for updates.";

		String headline1 = "Banksy 'arrested & real identity revealed' is the same hoax from last year";
		String headline3 = "BrFidel Castro is dead, according to viral Twitter rumors";
		String testTxt = "We have no information on whether users are at risk";
		/*
		 * for (List<String> s : trainingStances) {
		 * 
		 * }
		 */

		// Create the Stanford CoreNLP pipeline
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse,natlog,openie");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

		// StanfordDependencyParser sdp = new StanfordDependencyParser();
		/*
		 * List<SemanticGraph> graphs =
		 * StanfordDependencyParser.buildDependencyGraph(doc);
		 * 
		 * for (SemanticGraph graph : graphs) { System.out.println(graph); }
		 */
		Annotation doc = new Annotation(body);
		pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		List<SemanticGraph> graphs = StanfordDependencyParser.buildDependencyGraph(doc);
		// System.out.println("graphs size: " +graphs.size());
		// System.out.println("Sentences size: " + sentences.size());
		for (SemanticGraph graph : graphs) {
			System.out.println(graph);
			System.out.println(graph.toList());
			// System.out.println(graph.relns(graph.getNodeByIndexSafe(3)));
		}

		int i = 0;
		List<Map<String, String>> svos = new ArrayList<>();
		for (CoreMap sentence : sentences) {
			System.out.println("I'm here");
			System.out.println("sent. = " + sentence);
			// Get the OpenIE triples for the sentence
			Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
			// Print the triples
			System.out.println("triples= "+triples);
			System.out.println("sentence " + i + new Sentence(sentence).text());
			for (RelationTriple triple : triples) {
				System.out.println(triple.confidence + "\t" + "subj: " + triple.subjectLemmaGloss() + "\t" + "rel: "
						+ triple.relationLemmaGloss() + "\t" + "obj: " + triple.objectLemmaGloss() + "\t"
						+ "allTokens: " + triple.allTokens());

				List<CoreLabel> tokens = triple.allTokens();
				SemanticGraph relatedGraph = graphs.get(i);
	
				Map<String,String> vec = new HashMap<String, String>();
				for (CoreLabel t : tokens) {
					relatedGraph = graphs.get(i);
					int tIndex = t.index();
					System.out.println("tIdx= " + tIndex);
					System.out.println("relns = " + relatedGraph.relns(relatedGraph.getNodeByIndexSafe(tIndex)));

					Set<GrammaticalRelation> relns = relatedGraph.relns(relatedGraph.getNodeByIndexSafe(tIndex));
					for (GrammaticalRelation rel : relns) {
						if (rel.toString().equals("nsubj")) {
							vec.put("nsubj", t.word());
						}
						if (rel.toString().equals("dobj")) {
							vec.put("dobj", t.word());
						}
					}

				}
				
				if (vec.size() == 2){
					//find the verb
					String depList = relatedGraph.toList();
					String[] deps = depList.split("\n");
					for(String d : deps){
						String depType = d.substring(0, d.indexOf('('));
						System.out.println("depType = " + depType);
						if(depType.equals("nsubj")){
							String betweenBrack = d.substring(d.indexOf('(')+1, d.indexOf(')'));
						    String[] depWords = betweenBrack.split(",");
						    System.out.println("iiiiih = " + depWords[1].substring(0, depWords[1].indexOf('-')).trim());
						    System.out.println("kkkkk " + vec.get("nsubj"));
						    if(depWords[1].substring(0, depWords[1].indexOf('-')).trim().equals(vec.get("nsubj"))){
						    	vec.put("verb", depWords[0].substring(0, depWords[0].indexOf('-')).trim());
						    	System.out.println("Woooof  " + depWords[0].substring(0, depWords[0].indexOf('-')).trim());
						    }
						}
					}
					svos.add(vec);
					
				}

			}
			i++;
		}

		System.out.println("svos = " + svos);

		/*
		 * SemgrexPattern pattern = SemgrexPattern.compile(
		 * "{$}=root >/.subj(pass)?/ {}=subject >/.obj/ {}=object");
		 * SemgrexMatcher matcher = pattern.matcher(new Sentence(
		 * "A cat is sitting on the table").dependencyGraph()); while
		 * (matcher.find()) { IndexedWord root = matcher.getNode("root");
		 * IndexedWord subject = matcher.getNode("subject"); IndexedWord object
		 * = matcher.getNode("object"); System.out.println(root.word() + "(" +
		 * subject.word() + ", " + object.word()); }
		 */

	}
}
