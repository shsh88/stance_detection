package ude.master.thesis.stance_detection.util;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class TitleAndBodyTextPreprocess {

	public static final String[] SOURCE_WORDS = { "REPORT", "REPORTS", "PODCAST", "CNN", "CNBC", "Net Extra", "WSJ",
			"US Intel", "DHS", "BiH", "CNN audio", "BREAKING NEWS", "UPDATE", "source", "Pants on Fire", "BREAKING" };
	public static final char[] TO_REMOVE_QUOTS = { '′', '’', '‘', '“', '”' };
	public static final int NUM_SENT_BEG = 6;
	public static final int NUM_SENT_END = 4;
	private static Map<Integer, String> trainIdBodyMap;
	private static HashMap<Integer, String> testIdBodyMap;
	private static List<List<String>> testStances;
	private static List<List<String>> trainingStances;
	private static Map<Integer, String> trainIdBodyMapPreprocessed;
	private static HashMap<Integer, String> testIdBodyMapPreprocessed;
	private static StanfordCoreNLP pipeline;

	public static void main(String[] args) throws IOException {
		String body = "James Foley, an American journalist who went missing in Syria more than a year ago, "
				+ "has reportedly been executed by the Islamic State, a militant group formerly known as ISIS. "
				+ "Video and photos purportedly of Foley emerged on Tuesday. A YouTube video -- entitled \"\"A "
				+ "Message to #America (from the #IslamicState)\"\" -- identified a man on his knees as"
				+ " \"\"James Wright Foley,\"\" and showed his execution. This is a developing story. "
				+ "Check back here for updates.";
		// System.out.println(body);

		String txt = "Kim Jong-un has broken both of his ankles and is now in the hospital after undergoing "
				+ "surgery, a report in a South Korean newspaper claims. The North Korean leader has "
				+ "been missing for more than three weeks, fueling speculation about what could cause his "
				+ "unusual disappearance from the public eye. This rumor seems to confirm what North Korean "
				+ "state media had said on Thursday, when state broadcaster Korean Central Television "
				+ "reported that Kim was \"not feeling well,\" and was suffering from an \"uncomfortable "
				+ "physical condition.\" Have something to add to this story? Share it in the comments.";
		// System.out.println(txt);

		String testTxt1 = "The Islamic State (IS) leader Abu Bakr al-Baghdadi has not been killed as has "
				+ "been previously claimed. He is wounded and being treated in the border area of Iraq and "
				+ "Syria. A few days ago, it was reported that al-Baghdadi had been killed by a U.S. "
				+ "airstrike near Mosul in Northern Iraq, an attack that left three other senior members of "
				+ "the militant group dead. When it was reported that al-Baghdadi had been killed, the Pentagon "
				+ "did not confirm the death, but thousands of social media users shared an unverified photo "
				+ "claiming to be the ISIS leader’s body. However, Pentagon spokesman Col. Steve Warren did "
				+ "later say that any IS leaders “inside troop formations are likely to be killed.”";

		// System.out.println(testTxt1);

		String testTxt = "Newly released audio allegedly records the moment that Officer Darren Wilson "
				+ "opened fire on unarmed Michael Brown At least ten shots can be heard - in two separate "
				+ "volleys of gunfire Experts have ? said this : indicated a 'moment of contemplation' for Wilson "
				+ "FBI has confirmed it has interviewed the man who recorded audio "
				+ "Is another tantalizing piece of evidence collected in the ongoing case Officer Wilson "
				+ "claims he felt his life was threatened on August 9 Witnesses . . . and a "
				+ "friend of Brown, 1.8 , claim he had surrendered Brown was buried on Monday in a ceremony "
				+ "attended by thousands.  The FBI has been handed a potentially crucial recording that allegedly "
				+ "contains audio of the moment that Officer Darren Wilson opened fire and killed unarmed "
				+ "18-year-old Michael Brown in Ferguson   , Missouri  , earlier this month. Since the guard's "
				+ "arrival Monday, flare-ups in the small section of town that had been the center of nightly "
				+ "unrest have begun to subside  . About 100    ;   people gathered Thursday evening, walking in laps "
				+ "near the spot where Michael Brown was shot    .     Some were in organized groups, such as clergy "
				+ "members.More signs reflected calls by protesters to remove the prosecutor from the case.";
		// System.out.println(keepLimitedPunctuation(testTxt));

		String t1 = "Report: ISIS kidnaps Canadian-Israeli, former IDF soldier who went to fight with the Kurds";
		String t2 = "BREAKING NEWS: ISIS beheads missing American journalist James Wright Foley as warning to US to cease action in Iraq";
		String t3 = "James Foley executioner said identified as British rapper  Read more: James Foley executioner said identified as British rapper";
		String t4 = "Dog abandoned!! at Scottish rail... station with suitcase full?! of belongings  Read more: http://www.ctvnews.ca/world/dog-abandoned-at-scottish-rail-station-with-suitcase-full-of-belongings-1.2175026#ixzz3O9Shc1IO";

		System.out.println(fixQuotationMarks("Isis tells mother on rescue mission :’You’ve just eaten your son’"));
		// System.out.println(removeReadMoreWithFollowTxt(t3));
		// System.out.println(removeReadMoreWithFollowTxt(t4));
		/*
		 * System.out.println(testTxt1); testTxt1 =
		 * testTxt1.replaceAll("[^\\p{ASCII}]", ""); System.out.println(
		 * "After removing non ASCII chars:"); System.out.println(testTxt1);
		 * 
		 * String string = ".\"'"; System.out.println(testTxt1.replaceAll(
		 * "[\\p{P}&&[^\\.,\\?\\!\\:\\-\\']]", ""));
		 */

		// System.out.println(removeRepeatePunctuation(t4));
		/*
		 * System.out.println(removeNestedQuots(
		 * "US Central Command meanwhile admitted that the Islamic " +
		 * "State still had the ability 'to conduct small-scale operations, despite months "
		 * +
		 * "of air strikes. But, it said, their capacity to do so is degraded and their momentum is "
		 * + "stalling'."));
		 */
		/*
		 * String txt_ = "stalling'."; Pattern p = Pattern.compile("'[:\\.,;]");
		 * Matcher m = p.matcher(txt_); while (m.find()){ txt_ = m.replaceAll(""
		 * + m.group().charAt(1)); System.out.println(txt_); }
		 */

		/*
		 * System.out.println(removeUrl(
		 * "ritish intelligence officers estimate that there are around 500 homegrown"
		 * + " militants fighting for ISIS in Syria and Iraq. - See more at: " +
		 * "http://www.dailystar.com.lb/News/Middle-East/2015/Feb-26/288865-bbc-names-the-"
		 * +
		 * "jihadi-john-suspect-in-isis-beheading-videos-as-mohammed-emwazi-from-london.ashx#"
		 * + "sthash.4aTmHCWq.dpuf")); System.out.println(removeTwitterPicLinks(
		 * "British Intelligence has identified the ISIS killer of American, " +
		 * "Abdel Majed Bary of London, England! pic.twitter.com/FEEy1iHk0J— " +
		 * "DR. TWEET, PhD (@Callisto1947) August 24, 2014"));
		 */

		System.out.println(fixPunctuationSpacing(testTxt));
		/*
		 * StanceDetectionDataReader sddr = new StanceDetectionDataReader(true,
		 * true, "resources/data/train_stances.csv",
		 * "resources/data/train_bodies.csv",
		 * "resources/data/test_data/competition_test_stances.csv",
		 * "resources/data/test_data/competition_test_bodies.csv");
		 * 
		 * trainIdBodyMap = sddr.getTrainIdBodyMap(); trainingStances =
		 * sddr.getTrainStances();
		 * 
		 * testIdBodyMap = sddr.getTestIdBodyMap(); testStances =
		 * sddr.getTestStances();
		 */

		/*
		 * cleanTitles("resources/data/train_stances_preprocessed.csv",
		 * trainingStances);
		 * cleanTitles("resources/data/test_data/test_stances_preprocessed.csv",
		 * testStances);
		 * 
		 * cleanBodies("resources/data/train_bodies_preprocessed.csv",
		 * trainIdBodyMap);
		 * cleanBodies("resources/data/test_data/test_bodies_preprocessed.csv",
		 * testIdBodyMap);
		 */
		/*
		 * StanceDetectionDataReader sddr1 = new StanceDetectionDataReader(true,
		 * true, "resources/data/train_stances.csv",
		 * "resources/data/train_bodies_preprocessed.csv",
		 * "resources/data/test_data/competition_test_stances.csv",
		 * "resources/data/test_data/test_bodies_preprocessed.csv");
		 * trainIdBodyMapPreprocessed = sddr1.getTrainIdBodyMap();
		 * testIdBodyMapPreprocessed = sddr1.getTestIdBodyMap();
		 * 
		 * pipeline = getStanfordPipeline();
		 * summariseBody("resources/data/train_bodies_preprocessed_summ.csv",
		 * trainIdBodyMapPreprocessed); summariseBody(
		 * "resources/data/test_data/test_bodies_preprocessed_summ.csv",
		 * testIdBodyMapPreprocessed);
		 */
	}

	/**
	 * 
	 * @param csvFilePath
	 * @param trainIdBodyMapPreprocessed
	 * @throws IOException
	 */
	public static void summariseBody(String csvFilePath, Map<Integer, String> trainIdBodyMapPreprocessed)
			throws IOException {
		List<String[]> entries = new ArrayList<>();

		entries.add(new String[] { "Body ID", "sent_beg", "mid_body", "sent_end" });

		int i = 0;
		for (Map.Entry<Integer, String> e : trainIdBodyMapPreprocessed.entrySet()) {
			Annotation doc = new Annotation(e.getValue());
			pipeline.annotate(doc);
			List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

			System.out.println(sentences.size());
			List<CoreMap> usefulSentFromBeg = new ArrayList<>();
			List<CoreMap> usefulSentFromEnd = new ArrayList<>();
			List<CoreMap> middleBody = new ArrayList<>();
			int sIdx = 0;
			// Add sentences from the beginning of the article
			if (sentences.size() > NUM_SENT_BEG) {
				boolean found6 = false;

				for (CoreMap s : sentences) {
					if (s.toString().toLowerCase().contains("updated at"))
						System.out.println(s.toString());
					if ((s.toString().split(" ").length <= 2)) {
						System.out.println(s.toString());
					}

					if ((s.toString().split(" ").length > 2) && !(s.toString().toLowerCase().contains("updated at")))
						usefulSentFromBeg.add(s);
					if (usefulSentFromBeg.size() == NUM_SENT_BEG) {
						found6 = true;
						break;
					}
					sIdx++;
				}
				// if I could have 5 sentences from the beginning then I can
				// look at the rest of sentences
				int lIdx = sIdx;
				if (found6) {
					int k = 1;
					while (((sIdx - k) > 0) && usefulSentFromEnd.size() < NUM_SENT_END) {
						if ((sentences.size() - sIdx - k) > 0) {
							int size = sentences.size();
							// System.out.println("k= "+k);
							String sTxt = sentences.get(size - k).toString().toLowerCase();
							if (!(sTxt.contains("twitter") && sTxt.contains("follow "))
									&& !(sTxt.contains("facebook") && sTxt.contains("like "))
									&& !(sTxt.contains("read more")) && !(sTxt.contains("scroll down"))
									&& !(sTxt.contains("click here")) && !(sTxt.contains("click for"))
									&& !(sTxt.contains("click photo")) && !(sTxt.contains("updated at"))) {
								usefulSentFromEnd.add(sentences.get(size - k));
								lIdx = size - k;
							} else
								System.out.println(sentences.get(size - k));

						}
						k++;
					}
					// fill middle body if there are still text
					if (lIdx > sIdx) {
						for (int d = sIdx + 1; d < lIdx; d++) {
							String sTxt = sentences.get(d).toString().toLowerCase();
							if (!(sTxt.contains("twitter") && sTxt.contains("follow "))
									&& !(sTxt.contains("facebook") && sTxt.contains("like "))
									&& !(sTxt.contains("read more")) && !(sTxt.contains("scroll down"))
									&& !(sTxt.contains("click here")) && !(sTxt.contains("click for"))
									&& !(sTxt.contains("click photo")) && !(sTxt.contains("updated at"))) {
								middleBody.add(sentences.get(d));
							}

						}
					}
				}
			} else {
				for (CoreMap s : sentences) {
					usefulSentFromBeg.add(s);
				}
			}
			List<String> entry = new ArrayList<>();
			entry.add(String.valueOf(e.getKey()));

			if (usefulSentFromBeg.size() > 0) {
				String str = "";
				for (CoreMap u : usefulSentFromBeg)
					str += u.toString() + " ";
				entry.add(str.trim());
			} else
				entry.add("");

			if (middleBody.size() > 0) {
				String str = "";
				for (CoreMap u : middleBody)
					str += u.toString() + " ";
				entry.add(str.trim());
			} else
				entry.add("");

			if (usefulSentFromEnd.size() > 0) {
				String str = "";
				for (CoreMap u : usefulSentFromEnd)
					str += u.toString() + " ";
				entry.add(str.trim());
			} else
				entry.add("");

			entries.add(entry.toArray(new String[entry.size()]));

			i++;
			if (i % 3 == 50)
				System.out.println("processed: " + i);
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(csvFilePath))) {
			writer.writeAll(entries);
		}

	}

	public static StanfordCoreNLP getStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse,natlog,openie");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}

	/**
	 * 
	 * @param cleanedTitlesFilePath
	 * @param stances
	 * @throws IOException
	 */
	public static void cleanTitles(String cleanedTitlesFilePath, List<List<String>> stances) throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "Headline", "Body ID", "Stance" });

		for (List<String> s : stances) {
			String title = s.get(0);
			title = removeSpecificReportingWords(title);
			title = removeReadMoreWithFollowTxt(title);
			title = fixQuotationMarks(title);
			title = removeNonASCII(title);
			title = keepLimitedPunctuation(title);
			title = removeRepeatePunctuation(title);
			title = removeNestedQuots(title);
			title = removeUrl(title);
			title = replaceNewlineWithPeriod(title);
			title = fixPunctuationSpacing(title);

			List<String> entry = new ArrayList<>();
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			entries.add(entry.toArray(new String[entry.size()]));
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(cleanedTitlesFilePath))) {
			writer.writeAll(entries);
		}

	}

	/**
	 * 
	 * @param bodiesCsvFilePath
	 * @param trainIdBodyMap2
	 * @throws IOException
	 */
	public static void cleanBodies(String bodiesCsvFilePath, Map<Integer, String> trainIdBodyMap2) throws IOException {
		List<String[]> entries = new ArrayList<>();

		entries.add(new String[] { "Body ID", "articleBody" });
		int i = 0;
		for (Map.Entry<Integer, String> e : trainIdBodyMap2.entrySet()) {

			List<String> entry = new ArrayList<>();

			String body = e.getValue();
			body = fixQuotationMarks(body);
			body = removeNonASCII(body);
			body = keepLimitedPunctuation(body);
			body = removeRepeatePunctuation(body);
			body = removeNestedQuots(body);
			body = removeUrl(body);
			body = removeTwitterPicLinks(body);
			body = removeHTMLTags(body);
			body = replaceNewlineWithPeriod(body);
			body = fixPunctuationWithNoSpace(body);
			body = fixPunctuationSpacing(body);
			body = removeRepeatePunctuation(body);
			body = fixPunctuationSpacing(body);

			entry.add(Integer.toString(e.getKey()));
			entry.add(body);

			entries.add(entry.toArray(new String[entry.size()]));
			i++;

			// if (i == 20)
			// break;
			if (i % 1000 == 0)
				System.out.println("processed : " + i);
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(bodiesCsvFilePath))) {
			writer.writeAll(entries);
		}
	}

	// 1. Remove unprocessed articles
	// * Titles with one word
	// * non English text
	// *

	// 2. remove strings indicating article source
	/**
	 * use this just for the title
	 * 
	 * @param txt
	 * @return
	 */
	public static String removeSpecificReportingWords(String txt) {

		for (String s : SOURCE_WORDS) {
			Pattern p = Pattern.compile(s + ":", Pattern.CASE_INSENSITIVE);
			Matcher matcher = p.matcher(txt);
			while (matcher.find())
				txt = matcher.replaceAll("");
		}
		return txt.trim();
	}

	/**
	 * Use with titles only We may use these on bodies after getting sentences
	 * 
	 * @param txt
	 * @return
	 */
	public static String removeReadMoreWithFollowTxt(String txt) {
		int readMoreIdx = txt.toLowerCase().indexOf("read more");
		if (readMoreIdx > -1)
			txt = txt.substring(0, readMoreIdx);
		return txt;
	}

	// 3. Remove / replace with (') strange quotation marks
	/**
	 * do it for both title and body
	 * 
	 * @param txt
	 * @return
	 */
	public static String fixQuotationMarks(String txt) {
		for (char q : TO_REMOVE_QUOTS) {
			txt = txt.replace(q, '\'');
		}
		return txt;
	}

	// 4. Remove non ascii chars / may remove new line char
	public static String removeNonASCII(String txt) {
		txt = txt.replaceAll("[^\\p{ASCII}]", "");
		return txt;
	}

	// 5. Keep these punctuation ['?', ',', '.', ':', '-', '\''] and !
	public static String keepLimitedPunctuation(String txt) {
		txt = txt.replaceAll("[\\p{P}&&[^\\.,;\\?\\!\\:\\-\\']]", "");
		return txt;
	}

	// 6. Remove repeated punctuation
	public static String removeRepeatePunctuation(String txt) {
		txt = txt.replaceAll("(\\.\\.+)", ". ");
		txt = txt.replaceAll("(\\!\\!+)", "! ");
		txt = txt.replaceAll("(\\?\\?+)", "? ");
		txt = txt.replaceAll("(\\?\\!+)", "? ");
		return txt;
	}

	// 7. Remove nested quotation marks
	public static String removeNestedQuots(String txt) {
		txt = txt.replaceAll("^'", "");
		txt = txt.replaceAll("\\s'", " ");
		txt = txt.replaceAll("'\\s", " ");
		txt = txt.replaceAll("'$", "");

		Pattern p = Pattern.compile("[:\\.,;]'");
		Matcher m = p.matcher(txt);
		while (m.find())
			txt = m.replaceAll("" + m.group().charAt(0));

		p = Pattern.compile("'[:\\.,;]");
		Matcher m1 = p.matcher(txt);
		while (m1.find())
			txt = m1.replaceAll("" + m1.group().charAt(1));
		return txt;
	}

	public static String removeUrl(String txt) {
		String urlPattern = "((https?|ftp|gopher|telnet|file|Unsure|http):((//)|(\\\\))+[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*)";
		Pattern p = Pattern.compile(urlPattern, Pattern.CASE_INSENSITIVE);
		Matcher m = p.matcher(txt);
		int i = 0;
		while (m.find()) {
			txt = txt.replaceAll(m.group(i), "").trim();
			i++;
		}
		return txt;
	}

	public static String removeTwitterPicLinks(String txt) {
		String regex = "pic.twitter.com[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*";
		txt = txt.replaceAll(regex, "");
		return txt;
	}

	public static String removeHTMLTags(String txt) {
		txt = txt.replaceAll("<[^>]*>", "");
		return txt;
	}

	public static String fixPunctuationSpacing(String txt) {
		txt = txt.replaceAll("\\s+\\.\\s+", ". ");
		txt = txt.replaceAll("\\s+\\.", ". ");
		txt = txt.replaceAll("\\.\\s+", ". ");

		txt = txt.replaceAll("\\s+\\,\\s+", ", ");
		txt = txt.replaceAll("\\s+\\,", ", ");
		txt = txt.replaceAll("\\,\\s+", ", ");

		txt = txt.replaceAll("\\s+;\\s+", "; ");
		txt = txt.replaceAll("\\s+;", "; ");
		txt = txt.replaceAll(";\\s+", "; ");

		txt = txt.replaceAll("\\s+\\?\\s+", "; ");
		txt = txt.replaceAll("\\s+\\?", "; ");
		txt = txt.replaceAll("\\?\\s+", "; ");

		txt = txt.replaceAll("\\s+\\!\\s+", "! ");
		txt = txt.replaceAll("\\s+\\!", "! ");
		txt = txt.replaceAll("\\!\\s+", "! ");

		txt = txt.replaceAll("\\s+\\:\\s+", ": ");
		txt = txt.replaceAll("\\s+\\:", ": ");
		txt = txt.replaceAll("\\:\\s+", ": ");

		txt = txt.replaceAll("\\s+\\;\\s+", "; ");
		txt = txt.replaceAll("\\s+\\;", "; ");
		txt = txt.replaceAll("\\;\\s+", "; ");

		return txt;
	}

	private static String fixPunctuationWithNoSpace(String txt) {
		txt = txt.replaceAll("(\\D)(\\.)(\\D)", "$1" + ". " + "$3");
		txt = txt.replaceAll(",", ", ");
		txt = txt.replaceAll("\\?", "? ");
		txt = txt.replaceAll("\\!", "! ");
		txt = txt.replaceAll(";", "; ");
		txt = txt.replaceAll("(\\D)(:)(\\D)", ": ");
		return txt;
	}

	private static String fixPeriodsWithNumbers(String txt) {
		txt = txt.replaceAll("\\d\\.\\s+\\d", "$1" + ". " + "$3");
		return txt;
	}

	public static String replaceNewlineWithPeriod(String txt) {
		txt = txt.replaceAll("\\s+\\R", ". ");
		txt = txt.replaceAll("\\R\\s+", ". ");
		txt = txt.replaceAll("\\s+\\R\\s+", ". ");
		txt = txt.replaceAll("\\R", ". ");

		return txt;

	}
}
