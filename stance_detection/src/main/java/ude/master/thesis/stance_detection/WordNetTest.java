package ude.master.thesis.stance_detection;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.IPointer;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.ISynsetID;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import edu.mit.jwi.item.Pointer;
import edu.mit.jwi.morph.SimpleStemmer;
import edu.mit.jwi.morph.WordnetStemmer;

public class WordNetTest {

	public static void main(String[] args) throws IOException {
		// runExample();

		IDictionary dict = getDictionary();

		getHypernyms(dict);
	}

	public static void runExample() throws IOException {

		IDictionary dict = getDictionary();

		WordnetStemmer stemmer = new WordnetStemmer(dict);
		SimpleStemmer s = new SimpleStemmer();

		// look up first sense of the word "dog"
		IIndexWord idxWord = dict.getIndexWord("dog", POS.NOUN);

		List<String> synon = new ArrayList<>();
		List<String> anto = new ArrayList<>();

		for (int i = 0; i < idxWord.getWordIDs().size(); i++) {
			IWordID wordID = idxWord.getWordIDs().get(i);
			IWord word = dict.getWord(wordID);
			// System.out.println("Id = " + wordID);
			// System.out.println("Lemma = " + word.getLemma());
			// System.out.println("Gloss = " + word.getSynset().getGloss());

			System.out.println(word.getSynset());
			for (IWord s1 : word.getSynset().getWords()) {
				// System.out.println(s1.getLemma());
				synon.add(s1.getLemma());
			}

			Map<IPointer, List<IWordID>> rw = word.getRelatedMap();

			List<IWordID> ant = new ArrayList<>();
			for (Map.Entry<IPointer, List<IWordID>> rId : rw.entrySet()) {
				if (rId.getKey().toString().equals("antonym"))
					ant = rId.getValue();
			}

			// System.out.println("ANTONYMS");
			if (ant.size() > 0)
				for (IWordID id : ant) {
					IWord w = dict.getWord(id);
					// System.out.println(w.getLemma());
					anto.add(w.getLemma());
				}
		}
		// IWord w = new
		System.out.println("synonyms = " + synon);
		System.out.println("antonym = " + anto);
	}

	private static IDictionary getDictionary() throws IOException {
		// construct the URL to the Wordnet dictionary directory
		// String wnhome = System.getenv("WNHOME");
		// String path = wnhome + File.separator + "dict";
		String path = "C:/Master UDE/thesis/software/WordNet_3.1/dict";
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

	public static void getHypernyms(IDictionary dict) {
		// get the synset
		IIndexWord idxWord = dict.getIndexWord("dog", POS.NOUN);
		for (int j = 0; j < idxWord.getWordIDs().size(); j++) {
			IWordID wordID = idxWord.getWordIDs().get(j);
			IWord word = dict.getWord(wordID);
			ISynset synset = word.getSynset();

			// get the hypernyms
			List<ISynsetID> hypernyms = synset.getRelatedSynsets(Pointer.HYPERNYM_INSTANCE);

			// print out each hypernyms id and synonyms

			List<IWord> words;

			for (ISynsetID sid : hypernyms) {
				words = dict.getSynset(sid).getWords();
				System.out.print(sid + " {");
				for (Iterator<IWord> i = words.iterator(); i.hasNext();) {
					System.out.print(i.next().getLemma());
					if (i.hasNext())
						System.out.print(", ");
				}
				System.out.println("}");
			}
		}
	}

}
