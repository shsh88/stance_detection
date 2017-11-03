package ude.master.thesis.stance_detection.util;

import java.text.SimpleDateFormat;
import java.util.Date;

public class FNCConstants {

	public static final String ROOT_DIST_REFUTE = "root_dis_ref_";
	public static final String ROOT_DIST_DISCUSS = "root_dis_disc_";
	public static final String ROOT_DIST = "root_dis_";
	public static final String SVO = "svo_";
	public static final String PPDB_HUNG = "ppdb_";
	public static final String NEG_FEATURE = "neg";
	public static final String WORD2VEC_ADD_SIM = "w2v_sim_sum";
	public static final String TITLE_DOC2VEC = "t_d2vec_";
	public static final String BODY_DOC2VEC = "b_d2vec_";
	public static final String RELATION_NAME = "fnc-1";

	public static final String CLASS_ATTRIBUTE_NAME = "stance_class";

	public static final String ALL_STANCE_CLASSES[] = new String[] { "agree", "disagree", "discuss", "unrelated" };
	public static final String RELATED_STANCE_CLASSES[] = new String[] { "agree", "disagree", "discuss" };
	public static final String DISCUSS_STANCE_CLASSES[] = new String[] { "discuss", "non_discuss" };
	public static final String AGREE_STANCE_CLASSES[] = new String[] { "agree", "disagree" };
	public static final String BINARY_STANCE_CLASSES[] = new String[] { "related", "unrelated" };
	public static final String TEST = "_test";
	public static final String TRAIN = "_train";
	public static final String BODY_BOW_COUNTER = "body_Summ";
	public static final String WORD_OVERLAP = "word_overlap";
	public static final String REFUTE_FEATURE = "refute_";
	public static final String TITLE_POLARITY = "pol_title";
	public static final String BODY_POLARITY = "pol_body";
	public static final String BINARY_COOCCURANCE_COUNT = "bin_co_occ_count";
	public static final String BINARY_COOCCURANCE_COUNT_FIRST_255 = "bin_co_occ_255";
	public static final String BINARY_COOCCURANCE_STOP_COUNT = "bin_co_occ_stop_count";
	public static final String BINARY_COOCCURANCE_STOP_COUNT_FIRST_255 = "bin_co_occ_stop_255";
	public static final String NGRAM_HITS = "ngram_hits_";
	public static final String NGRAM_EARLY_HITS = "ngram_early_hits_";
	public static final String CHAR_GRAMS_HITS = "cgram_hits_";
	public static final String CHAR_GRAMS_EARLY_HITS = "cgram_early_hits_";
	public static final String CHAR_GRAMS_FIRST_HITS = "cgram_first_hits_";
	public static final String COSINE_METRIC_SIM = "cos_sim";
	public static final String LESK_OVERLAP = "lesk";
	public static final String TITLE_Q = "Q";
	public static final String HYP_SIM = "hyp_sim";
	public static final String TITLE_LENGTH = "t_len";
	public static final String BODY_SUMM_BOW_COUNTER = "bow_summ";
	public static final String NEG_FEATURE_ARG = "neg_arg";
	public static final String WORD2VEC_MLULT_SIM = "w2v_mult_sim";
	public static final String PUNC_COUNT = "punc_count";
	public static final String ARGS_COUNT = "args_count";
	public static final String TITLE_SENTIMENT = "t_senti";
	public static final String BODY_SENTIMENTS = "b_senti";
	public static final String PPDB_TLDR_HUNG = "ppdb_tldr";
	public static final String NEG_TLDR_FEATURE = "neg_tldr";
	public static final String TITLE_BIAS_COUNT = "t_bias_count";
	public static final String BODY_BIAS_COUNT = "b_bias_count";
	public static final String BODY_IDF = "idf_";
	public static final String SENTENCE_LENGTH_AVG = "sent_avg";
	public static final String SENTENCE_LENGTH_MAX = "sent_max";
	public static final String POS_TAGs_STR = "pos_tags_str";

	public static String getCurrentTimeStamp() {
		return new SimpleDateFormat("MM-dd_HH-mm").format(new Date());
	}

}
