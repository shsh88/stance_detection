package ude.master.thesis.stance_detection.processor;

import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.CoreMap;

public class StanfordParserTest {

	public Annotation parse(String text) {
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

	public void buildDependencyGraph(Annotation document) {
		// these are all the sentences in this document
		// a CoreMap is essentially a Map that uses class objects as keys and
		// has values with custom types
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		

		for (CoreMap sentence : sentences) {
			// traversing the words in the current sentence
			// a CoreLabel is a CoreMap with additional token-specific methods
			/*for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// this is the text of the token
				String word = token.get(TextAnnotation.class);
				// this is the POS tag of the token
				String pos = token.get(PartOfSpeechAnnotation.class);
				// this is the NER label of the token
				String ne = token.get(NamedEntityTagAnnotation.class);
				
				int idx = token.get(IndexAnnotation.class);
				System.out.println(idx);
			}*/

			// this is the parse tree of the current sentence
			//Tree tree = sentence.get(TreeAnnotation.class);
			//System.out.println(tree);

			// this is the Stanford dependency graph of the current sentence
			SemanticGraph dependencies = sentence.get(EnhancedDependenciesAnnotation.class);
			String dep_type = "EnhancedDependenciesAnnotation";
			System.out.println(dep_type+" ===>>");
			System.out.println("Sentence: "+sentence.toString());
			System.out.println("DEPENDENCIES: "+dependencies.toList());
			System.out.println("DEPENDENCIES SIZE: "+dependencies.size());
			Iterable<SemanticGraphEdge> edge_set = dependencies.edgeIterable();
			
			int j = 0;
			for(SemanticGraphEdge edge : edge_set){
			    j++;
			    System.out.println("------EDGE DEPENDENCY: "+j);
			    Iterator<SemanticGraphEdge> it = edge_set.iterator();
			    IndexedWord dep = edge.getDependent();
			    String dependent = dep.word();
			    int dependent_index = dep.index();
			    IndexedWord gov = edge.getGovernor();
			    String governor = gov.word();
			    int governor_index = gov.index();
			    GrammaticalRelation relation = edge.getRelation();
			    System.out.println("No:"+j+" Relation: "+relation.toString()+" Dependent ID: "+dep.index()+" Dependent: "+dependent.toString()+" Governor ID: "+gov.index()+" Governor: "+governor.toString());
			}
			
			System.out.println("====  Roots ====");
			System.out.println(dependencies.getRoots());
			
			System.out.println(dependencies.getShortestDirectedPathEdges(dependencies.getFirstRoot(), dependencies.getNodeByIndex(6)));

		}
	}

	public static void main(String[] args) {
		StanfordParserTest sp = new StanfordParserTest();
		
		String text = "(NEWSER) – Wonder how long a Quarter Pounder with cheese can "
				+ "last? Two Australians say they bought a few McDonald's burgers for "
				+ "friends back in 1995, when they were teens, and one of the friends "
				+ "never showed up. So the kid's burger went uneaten—and stayed that way, "
				+ "Australia's News Network reports. \"We’re pretty sure it’s the oldest"
				+ " burger in the world,\" says one of the men, Casey Dean. Holding onto "
				+ "the burger for their friend \"started off as a joke,\" he adds, but \""
				+ "the months became years and now, 20 years later, it looks the same"
				+ " as it did the day we bought it, perfectly preserved in its original "
				+ "wrapping.\" Dean and his burger-buying mate, Eduard Nitz, even took "
				+ "the burger on Australian TV show The Project last night and \"showed"
				+ " off the mold-free specimen,\" News 9 reports. The pair offered to "
				+ "take a bite of it for charity but were dissuaded by the show's hosts. "
				+ "They've also started a Facebook page for the burger called \"Can This "
				+ "20 Year Old Burger Get More Likes Than Kanye West?\" that has more than "
				+ "4,044 likes as of this writing. And they're selling an iTunes song, "
				+ "\"Free the Burger,\" for $1.69, and giving proceeds to the charity "
				+ "Beyond Blue, which helps Australians battle anxiety and depression. "
				+ "(A few years ago, a man didn't sell a 20-year-old bottle of McDonald's "
				+ "McJordan sauce for $10,000. Here's why Mickey D's food seemingly " + "never decays.)";
		
		String text2 = "BMO forecasts 19M Apple Watch sales in 2015, with more than half selling in holiday season";
		String text3 = "Iraq Says Arrested Woman Is Not The Wife of ISIS Leader al-Baghdadi";
		Annotation doc = sp.parse(text3);
		sp.buildDependencyGraph(doc);
		

	}

}
