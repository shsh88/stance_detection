package ude.master.thesis.stance_detection.util;

import java.text.SimpleDateFormat;
import java.util.Date;

public class FNCConstants {

	public static final String ROOT_DIST_REFUTE = "root_dis_ref_";
	public static final String ROOT_DIST_DISCUSS = "root_dis_disc_";
	public static final String SVO = "svo_";
	public static final String PPDB_HUNG = "ppdb_";
	public static final String NEG_FEATURE = "neg";
	public static final String WORD2VEC_ADD_SIMM = "w2v_sim_sum";
	public static final String TITLE_DOC2VEC = "t_d2vec_";
	public static final String BODY_DOC2VEC = "b_d2vec_";
	public static final String RELATION_NAME = "fnc-1";
	
	public static final String CLASS_ATTRIBUTE_NAME = "stance_class";

	public static final String ALL_STANCE_CLASSES[] = new String[] { "agree", "disagree", "discuss", "unrelated" };
	public static final String RELATED_STANCE_CLASSES[] = new String[] { "agree", "disagree", "discuss"};
	public static final String BINARY_STANCE_CLASSES[] = new String[] { "related", "unrelated" };
	public static final String TEST = "_tset";
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
	
	
	public static String getCurrentTimeStamp() {
		return new SimpleDateFormat("MM-dd_HH-mm").format(new Date());
	}
	
}
