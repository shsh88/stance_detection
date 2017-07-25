package ude.master.thesis.stance_detection;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;

/**
 * Hello world!
 *
 */
public class App {

	public static List<String> tokenizeString(Analyzer analyzer, String string) {
		List<String> result = new ArrayList<String>();
		try {
			TokenStream stream = analyzer.tokenStream(null, new StringReader(string));
			stream.reset();
			while (stream.incrementToken()) {
				result.add(stream.getAttribute(CharTermAttribute.class).toString());
			}
			stream.reset();
			stream.end();
			stream.close();
		} catch (IOException e) {
			// not thrown b/c we're using a string reader...
			throw new RuntimeException(e);
		}
		
		return result;
	}

	public static void main(String[] args) {
		System.out.println("Hello World!");
		
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
				+ "McJordan sauce for $10,000. Here's why Mickey D's food seemingly "
				+ "never decays.)";
		System.out.println(tokenizeString(new StandardAnalyzer(), "the boy's cars are different colors in the sun"));
		
	}
}
