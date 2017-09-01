package ude.master.thesis.stance_detection.processor;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * 
 * @author Razan
 *
 */
public class Lemmatizer {

	private StanfordCoreNLP pipeline;
	private Porter porter;

	public Lemmatizer() {
		// Create StanfordCoreNLP object properties, with POS tagging
		// (required for lemmatization), and lemmatization
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma");

		this.pipeline = new StanfordCoreNLP(props);
	}

	/**
	 * Method to perform lemmatization on a given text
	 * 
	 * @param documentText
	 *            the text to lemmatize
	 * @return a List with all the words' lemmas
	 */
	public List<String> lemmatize(String documentText) {
		List<String> lemmas = new ArrayList<>();
		// Create an empty Annotation just with the given text
		Annotation document = new Annotation(documentText);
		// run all Annotators on this text
		this.pipeline.annotate(document);
		// Iterate over all of the sentences found
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
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
	
	/**
	 * Get the lemmas with indexes after removing stop words
	 * @param documentText
	 * @return
	 */
	public Map<String, Integer> lemmatizeWithIdx(String documentText) {
		if(porter==null)
			porter = new Porter();
		//String cleanTxt = FeatureExtractor.clean(documentText);
		
		Map<String, Integer> lemmas = new HashMap<>();
		// Create an empty Annotation just with the given text
		Annotation document = new Annotation(documentText);
		// run all Annotators on this text
		this.pipeline.annotate(document);
		// Iterate over all of the sentences found
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		int i = 0; //for sentences
		int j = 0; //for tokens
		//System.out.println("ssize " + sentences.size());
		for (CoreMap sentence : sentences) {
			// Iterate over all tokens in a sentence
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// Retrieve and add the lemma for each word into the
				// list of lemmas
				String lemma = token.get(LemmaAnnotation.class);
				lemma = FeatureExtractor.clean(lemma).trim();
				
				try {
					if(FeatureExtractor.isStopword(lemma) || lemma.isEmpty())
						continue;
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				lemmas.put(""+ porter.stripAffixes(lemma) + ","+ i + ","+ token.index(), j);
				j++;
			}
			i++;
		}
		return lemmas;
	}

	public static void main(String[] args) {
		Lemmatizer lemm = new Lemmatizer();

		String str = "(NEWSER) – Wonder how long a Quarter Pounder with cheese can "
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
				+ "(A few years ago, a man sold a 20-year-old bottle of McDonald's "
				+ "McJordan sauce for $10,000. Here's why Mickey D's food seemingly " + "never decays.)";
		String resultString = str.replaceAll("\\W", " ").toLowerCase();
		// String resultString = cleanText(str);
		System.out.println(lemm.lemmatize(resultString));
	}
}
