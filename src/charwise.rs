//! A character-wise version for faster matching on multibyte characters.
//!
//! This sub-module provides a character-wise implementation of Daachorse,
//! [`CharwiseDoubleArrayAhoCorasick`].
//! The standard version [`DoubleArrayAhoCorasick`](super::DoubleArrayAhoCorasick)
//! handles strings as UTF-8 sequences
//! and defines transition labels using byte integers.
//! On the other hand, the character-wise version uses code point values of Unicode,
//! resulting in reducing the number of transitions and faster matching on multibyte characters.
//!
//! # Features
//!
//! Compared to [`DoubleArrayAhoCorasick`](super::DoubleArrayAhoCorasick),
//! [`CharwiseDoubleArrayAhoCorasick`] has the following features
//! if it is built from multibyte strings such as Japanese:
//!
//!  - Faster matching can be expected.
//!  - The construction time can be slower.
//!  - The memory efficiency depends on input patterns.
//!    - If the scale is large, the memory efficiency can be competitive.
//!    - If the scale is small, the double array can be sparse and memory inefficiency.
//!
//! # Examples
//!
//! The example finds non-overlapped occurrences with shortest matching
//! on UTF-8 strings.
//! Note that byte positions are reported.
//!
//! ```rust
//! use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
//!
//! let patterns = vec!["全世界", "世界", "に"];
//! let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
//!
//! let mut it = pma.find_iter("全世界中に");
//!
//! let m = it.next().unwrap();
//! assert_eq!((0, 9, 0), (m.start(), m.end(), m.value()));
//!
//! let m = it.next().unwrap();
//! assert_eq!((12, 15, 2), (m.start(), m.end(), m.value()));
//!
//! assert_eq!(None, it.next());
//! ```
pub mod builder;
pub mod iter;

use std::io::{self, Read, Write};

pub use crate::charwise::builder::CharwiseDoubleArrayAhoCorasickBuilder;
use crate::charwise::iter::{
    CharWithEndOffsetIterator, FindIterator, FindOverlappingIterator,
    FindOverlappingNoSuffixIterator, LestmostFindIterator, StrIterator,
};
use crate::errors::Result;
use crate::{MatchKind, Output};
use crate::utils;

// The maximum BASE value used as an invalid value.
pub(crate) const BASE_INVALID: i32 = std::i32::MAX;
// The maximum output position value used as an invalid value.
pub(crate) const OUTPUT_POS_INVALID: u32 = std::u32::MAX;
// The root index position.
pub(crate) const ROOT_STATE_IDX: u32 = 0;
// The dead index position.
pub(crate) const DEAD_STATE_IDX: u32 = 1;

/// Fast multiple pattern match automaton implemented
/// with the Aho-Corasick algorithm and character-wise double-array data structure.
///
/// # Build instructions
///
/// [`CharwiseDoubleArrayAhoCorasick`] supports the following two types of input data:
///
/// - [`CharwiseDoubleArrayAhoCorasick::new`] builds an automaton from a set of UTF-8 strings
///    while assigning unique identifiers in the input order.
///
/// - [`CharwiseDoubleArrayAhoCorasick::with_values`] builds an automaton
///    from a set of pairs of a UTF-8 string and a `u32` value.
#[derive(Clone)]
pub struct CharwiseDoubleArrayAhoCorasick {
    states: Vec<State>,
    outputs: Vec<Output>,
    match_kind: MatchKind,
    num_states: u32,
}

impl CharwiseDoubleArrayAhoCorasick {
    /// Creates a new [`CharwiseDoubleArrayAhoCorasick`] from input patterns.
    /// The value `i` is automatically associated with `patterns[i]`.
    ///
    /// # Arguments
    ///
    /// * `patterns` - List of patterns.
    ///
    /// # Errors
    ///
    /// [`DaachorseError`](super::errors::DaachorseError) is returned when
    ///   - `patterns` is empty,
    ///   - `patterns` contains entries of length zero,
    ///   - `patterns` contains duplicate entries,
    ///   - the scale of `patterns` exceeds the expected one, or
    ///   - the scale of the resulting automaton exceeds the expected one.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["全世界", "世界", "に"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// let mut it = pma.find_iter("全世界中に");
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 9, 0), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((12, 15, 2), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    pub fn new<I, P>(patterns: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        CharwiseDoubleArrayAhoCorasickBuilder::new().build(patterns)
    }

    /// Creates a new [`CharwiseDoubleArrayAhoCorasick`] from input pattern-value pairs.
    ///
    /// # Arguments
    ///
    /// * `patvals` - List of pattern-value pairs, in which the value is of type `u32` and less than `u32::MAX`.
    ///
    /// # Errors
    ///
    /// [`DaachorseError`](super::errors::DaachorseError) is returned when
    ///   - `patvals` is empty,
    ///   - `patvals` contains patterns of length zero,
    ///   - `patvals` contains duplicate patterns,
    ///   - `patvals` contains invalid values,
    ///   - the scale of `patvals` exceeds the expected one, or
    ///   - the scale of the resulting automaton exceeds the expected one.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patvals = vec![("全世界", 0), ("世界", 10), ("に", 100)];
    /// let pma = CharwiseDoubleArrayAhoCorasick::with_values(patvals).unwrap();
    ///
    /// let mut it = pma.find_iter("全世界中に");
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 9, 0), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((12, 15, 100), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    pub fn with_values<I, P>(patvals: I) -> Result<Self>
    where
        I: IntoIterator<Item = (P, u32)>,
        P: AsRef<str>,
    {
        CharwiseDoubleArrayAhoCorasickBuilder::new().build_with_values(patvals)
    }

    /// Returns an iterator of non-overlapping matches in the given haystack.
    ///
    /// # Arguments
    ///
    /// * `haystack` - String to search for.
    ///
    /// # Panics
    ///
    /// When you specify `MatchKind::{LeftmostFirst,LeftmostLongest}` in the construction,
    /// the iterator is not supported and the function will call panic!.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["全世界", "世界", "に"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// let mut it = pma.find_iter("全世界中に");
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 9, 0), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((12, 15, 2), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    pub fn find_iter<P>(&self, haystack: P) -> FindIterator<StrIterator<P>>
    where
        P: AsRef<str>,
    {
        assert!(
            self.match_kind.is_standard(),
            "Error: match_kind must be standard."
        );
        FindIterator {
            pma: self,
            haystack: unsafe { CharWithEndOffsetIterator::new(StrIterator::new(haystack)) },
        }
    }

    /// Returns an iterator of non-overlapping matches in the given haystack iterator.
    ///
    /// # Arguments
    ///
    /// * `haystack` - String to search for.
    ///
    /// # Panics
    ///
    /// When you specify `MatchKind::{LeftmostFirst,LeftmostLongest}` in the construction,
    /// the iterator is not supported and the function will call panic!.
    ///
    /// # Safety
    ///
    /// `haystack` must represent a valid UTF-8 string.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["全世界", "世界", "に"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// let haystack = "全世界".as_bytes().iter().chain("中に".as_bytes()).copied();
    ///
    /// let mut it = unsafe { pma.find_iter_from_iter(haystack) };
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 9, 0), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((12, 15, 2), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    pub unsafe fn find_iter_from_iter<P>(&self, haystack: P) -> FindIterator<P>
    where
        P: Iterator<Item = u8>,
    {
        assert!(
            self.match_kind.is_standard(),
            "Error: match_kind must be standard."
        );
        FindIterator {
            pma: self,
            haystack: CharWithEndOffsetIterator::new(haystack),
        }
    }

    /// Returns an iterator of overlapping matches in the given haystack.
    ///
    /// # Arguments
    ///
    /// * `haystack` - String to search for.
    ///
    /// # Panics
    ///
    /// When you specify `MatchKind::{LeftmostFirst,LeftmostLongest}` in the construction,
    /// the iterator is not supported and the function will call panic!.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["全世界", "世界", "に"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// let mut it = pma.find_overlapping_iter("全世界中に");
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 9, 0), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((3, 9, 1), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((12, 15, 2), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    pub fn find_overlapping_iter<P>(&self, haystack: P) -> FindOverlappingIterator<StrIterator<P>>
    where
        P: AsRef<str>,
    {
        assert!(
            self.match_kind.is_standard(),
            "Error: match_kind must be standard."
        );
        FindOverlappingIterator {
            pma: self,
            haystack: unsafe { CharWithEndOffsetIterator::new(StrIterator::new(haystack)) },
            state_id: ROOT_STATE_IDX,
            pos: 0,
            output_pos: 0,
        }
    }

    /// Returns an iterator of overlapping matches in the given haystack iterator.
    ///
    /// # Arguments
    ///
    /// * `haystack` - String to search for.
    ///
    /// # Panics
    ///
    /// When you specify `MatchKind::{LeftmostFirst,LeftmostLongest}` in the construction,
    /// the iterator is not supported and the function will call panic!.
    ///
    /// # Safety
    ///
    /// `haystack` must represent a valid UTF-8 string.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["全世界", "世界", "に"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// let haystack = "全世界".as_bytes().iter().chain("中に".as_bytes()).copied();
    ///
    /// let mut it = unsafe { pma.find_overlapping_iter_from_iter(haystack) };
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 9, 0), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((3, 9, 1), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((12, 15, 2), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    pub unsafe fn find_overlapping_iter_from_iter<P>(
        &self,
        haystack: P,
    ) -> FindOverlappingIterator<P>
    where
        P: Iterator<Item = u8>,
    {
        assert!(
            self.match_kind.is_standard(),
            "Error: match_kind must be standard."
        );
        FindOverlappingIterator {
            pma: self,
            haystack: CharWithEndOffsetIterator::new(haystack),
            state_id: ROOT_STATE_IDX,
            pos: 0,
            output_pos: 0,
        }
    }

    /// Returns an iterator of overlapping matches without suffixes in the given haystack.
    ///
    /// The Aho-Corasick algorithm reads through the haystack from left to right and reports
    /// matches when it reaches the end of each pattern. In the overlapping match, more than one
    /// pattern can be returned per report.
    ///
    /// This iterator returns the first match on each report.
    ///
    /// # Arguments
    ///
    /// * `haystack` - String to search for.
    ///
    /// # Panics
    ///
    /// When you specify `MatchKind::{LeftmostFirst,LeftmostLongest}` in the construction,
    /// the iterator is not supported and the function will call panic!.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["全世界", "世界", "に"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// let mut it = pma.find_overlapping_no_suffix_iter("全世界中に");
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 9, 0), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((12, 15, 2), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    pub fn find_overlapping_no_suffix_iter<P>(
        &self,
        haystack: P,
    ) -> FindOverlappingNoSuffixIterator<StrIterator<P>>
    where
        P: AsRef<str>,
    {
        assert!(
            self.match_kind.is_standard(),
            "Error: match_kind must be standard."
        );
        FindOverlappingNoSuffixIterator {
            pma: self,
            haystack: unsafe { CharWithEndOffsetIterator::new(StrIterator::new(haystack)) },
            state_id: ROOT_STATE_IDX,
        }
    }

    /// Returns an iterator of overlapping matches without suffixes in the given haystack iterator.
    ///
    /// The Aho-Corasick algorithm reads through the haystack from left to right and reports
    /// matches when it reaches the end of each pattern. In the overlapping match, more than one
    /// pattern can be returned per report.
    ///
    /// This iterator returns the first match on each report.
    ///
    /// # Arguments
    ///
    /// * `haystack` - String to search for.
    ///
    /// # Panics
    ///
    /// When you specify `MatchKind::{LeftmostFirst,LeftmostLongest}` in the construction,
    /// the iterator is not supported and the function will call panic!.
    ///
    /// # Safety
    ///
    /// `haystack` must represent a valid UTF-8 string.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["全世界", "世界", "に"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// let haystack = "全世界".as_bytes().iter().chain("中に".as_bytes()).copied();
    ///
    /// let mut it = unsafe { pma.find_overlapping_no_suffix_iter_from_iter(haystack) };
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 9, 0), (m.start(), m.end(), m.value()));
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((12, 15, 2), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    pub unsafe fn find_overlapping_no_suffix_iter_from_iter<P>(
        &self,
        haystack: P,
    ) -> FindOverlappingNoSuffixIterator<P>
    where
        P: Iterator<Item = u8>,
    {
        assert!(
            self.match_kind.is_standard(),
            "Error: match_kind must be standard."
        );
        FindOverlappingNoSuffixIterator {
            pma: self,
            haystack: CharWithEndOffsetIterator::new(haystack),
            state_id: ROOT_STATE_IDX,
        }
    }

    /// Returns an iterator of leftmost matches in the given haystack.
    ///
    /// The leftmost match greedily searches the longest possible match at each iteration, and
    /// the match results do not overlap positionally such as [`CharwiseDoubleArrayAhoCorasick::find_iter()`].
    ///
    /// According to the [`MatchKind`] option you specified in the construction,
    /// the behavior is changed for multiple possible matches, as follows.
    ///
    ///  - If you set [`MatchKind::LeftmostLongest`], it reports the match
    ///    corresponding to the longest pattern.
    ///
    ///  - If you set [`MatchKind::LeftmostFirst`], it reports the match
    ///    corresponding to the pattern earlier registered to the automaton.
    ///
    /// # Arguments
    ///
    /// * `haystack` - String to search for.
    ///
    /// # Panics
    ///
    /// When you do not specify `MatchKind::{LeftmostFirst,LeftmostLongest}` in the construction,
    /// the iterator is not supported and the function will call panic!.
    ///
    /// # Examples
    ///
    /// ## LeftmostLongest
    ///
    /// ```
    /// use daachorse::MatchKind;
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasickBuilder;
    ///
    /// let patterns = vec!["世界", "世", "世界中に"];
    /// let pma = CharwiseDoubleArrayAhoCorasickBuilder::new()
    ///           .match_kind(MatchKind::LeftmostLongest)
    ///           .build(&patterns)
    ///           .unwrap();
    ///
    /// let mut it = pma.leftmost_find_iter("世界中に");
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 12, 2), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    ///
    /// ## LeftmostFirst
    ///
    /// ```
    /// use daachorse::MatchKind;
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasickBuilder;
    ///
    /// let patterns = vec!["世界", "世", "世界中に"];
    /// let pma = CharwiseDoubleArrayAhoCorasickBuilder::new()
    ///           .match_kind(MatchKind::LeftmostFirst)
    ///           .build(&patterns)
    ///           .unwrap();
    ///
    /// let mut it = pma.leftmost_find_iter("世界中に");
    ///
    /// let m = it.next().unwrap();
    /// assert_eq!((0, 6, 0), (m.start(), m.end(), m.value()));
    ///
    /// assert_eq!(None, it.next());
    /// ```
    pub fn leftmost_find_iter<P>(&self, haystack: P) -> LestmostFindIterator<P>
    where
        P: AsRef<str>,
    {
        assert!(
            self.match_kind.is_leftmost(),
            "Error: match_kind must be leftmost."
        );
        LestmostFindIterator {
            pma: self,
            haystack,
            pos: 0,
        }
    }

    /// Returns the total number of states this automaton has.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["bcd", "ab", "a"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// assert_eq!(pma.num_states(), 6);
    /// ```
    pub const fn num_states(&self) -> u32 {
        self.num_states
    }

    /// Returns the total number of elements of the double array.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["bcd", "ab", "a"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// assert_eq!(pma.num_elements(), 7);
    /// ```
    pub fn num_elements(&self) -> usize {
        self.states.len()
    }

    /// Returns the total amount of heap used by this automaton in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
    ///
    /// let patterns = vec!["bcd", "ab", "a"];
    /// let pma = CharwiseDoubleArrayAhoCorasick::new(patterns).unwrap();
    ///
    /// assert_eq!(pma.heap_bytes(), 144);
    /// ```
    pub fn heap_bytes(&self) -> usize {
        self.states.len() * std::mem::size_of::<State>()
            + self.outputs.len() * std::mem::size_of::<Output>()
    }

    pub fn serialize<W>(&self, mut wtr: W) -> io::Result<()> where W: Write {
        utils::write_u32(&mut wtr, self.states.len() as u32)?;
        for state in &self.states {
            state.serialize(&mut wtr)?;
        }
        utils::write_u32(&mut wtr, self.outputs.len() as u32)?;
        for output in &self.outputs {
            output.serialize(&mut wtr)?;
        }
        utils::write_u8(&mut wtr, self.match_kind as u8)?;
        utils::write_u32(&mut wtr, self.num_states)?;
        Ok(())
    }

    pub fn deserialize<R>(mut rdr: R) -> io::Result<Self> where R: Read {
        let states_len = utils::read_u32(&mut rdr)? as usize;
        let mut states = Vec::with_capacity(states_len);
        for _ in 0..states_len {
            states.push(State::deserialize(&mut rdr)?);
        }
        let outputs_len = utils::read_u32(&mut rdr)? as usize;
        let mut outputs = Vec::with_capacity(outputs_len);
        for _ in 0..outputs_len {
            outputs.push(Output::deserialize(&mut rdr)?);
        }
        let match_kind = MatchKind::from(utils::read_u8(&mut rdr)?);
        let num_states = utils::read_u32(&mut rdr)?;

        if !validate(&states, &outputs, match_kind.is_leftmost()) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid automaton"));
        }

        Ok(Self {
            states,
            outputs,
            match_kind,
            num_states,
        })
    }

    /// # Safety
    ///
    /// `state_id` must be smaller than the length of states.
    #[inline(always)]
    unsafe fn get_child_index_unchecked(&self, state_id: u32, mapped_c: u32) -> Option<u32> {
        if let Some(base) = self.states.get_unchecked(state_id as usize).base() {
            let child_idx = base + mapped_c as i32;
            if child_idx < 0 {
                return None;
            }
            // Use `get()` since `child_idx` can be no less than `self.states.len()`.
            if let Some(child) = self.states.get(child_idx as usize) {
                if child.check() == state_id {
                    return Some(child_idx as u32);
                }
            }
        }
        None
    }

    /// # Safety
    ///
    /// `state_id` must be smaller than the length of states.
    #[inline(always)]
    unsafe fn get_next_state_id_unchecked(&self, mut state_id: u32, mapped_c: u32) -> u32 {
        loop {
            if let Some(state_id) = self.get_child_index_unchecked(state_id, mapped_c) {
                return state_id;
            }
            if state_id == ROOT_STATE_IDX {
                return ROOT_STATE_IDX;
            }
            state_id = self.states.get_unchecked(state_id as usize).fail();
        }
    }

    /// # Safety
    ///
    /// `state_id` must be smaller than the length of states.
    #[inline(always)]
    unsafe fn get_next_state_id_leftmost_unchecked(&self, mut state_id: u32, mapped_c: u32) -> u32 {
        loop {
            if let Some(state_id) = self.get_child_index_unchecked(state_id, mapped_c) {
                return state_id;
            }
            if state_id == ROOT_STATE_IDX {
                return ROOT_STATE_IDX;
            }
            let fail_id = self.states.get_unchecked(state_id as usize).fail();
            if fail_id == DEAD_STATE_IDX {
                return DEAD_STATE_IDX;
            }
            state_id = fail_id;
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct State {
    base: i32,
    check: u32,
    fail: u32,
    output_pos: u32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            base: BASE_INVALID,
            check: DEAD_STATE_IDX,
            fail: DEAD_STATE_IDX,
            output_pos: OUTPUT_POS_INVALID,
        }
    }
}

impl State {
    #[inline(always)]
    pub fn base(&self) -> Option<i32> {
        Some(self.base).filter(|&x| x != BASE_INVALID)
    }

    #[inline(always)]
    pub const fn check(&self) -> u32 {
        self.check
    }

    #[inline(always)]
    pub const fn fail(&self) -> u32 {
        self.fail
    }

    #[inline(always)]
    pub fn output_pos(&self) -> Option<u32> {
        Some(self.output_pos).filter(|&x| x != OUTPUT_POS_INVALID)
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn set_base(&mut self, x: i32) {
        self.base = x;
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn set_check(&mut self, x: u32) {
        self.check = x;
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn set_fail(&mut self, x: u32) {
        self.fail = x;
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn set_output_pos(&mut self, x: u32) {
        self.output_pos = x;
    }

    #[inline(always)]
    fn serialize<W>(&self, mut wtr: W) -> io::Result<()> where W: Write {
        utils::write_i32(&mut wtr, self.base)?;
        utils::write_u32(&mut wtr, self.check)?;
        utils::write_u32(&mut wtr, self.fail)?;
        utils::write_u32(&mut wtr, self.output_pos)?;
        Ok(())
    }

    #[inline(always)]
    fn deserialize<R>(mut rdr: R) -> io::Result<Self> where R: Read {
        Ok(Self {
            base: utils::read_i32(&mut rdr)?,
            check: utils::read_u32(&mut rdr)?,
            fail: utils::read_u32(&mut rdr)?,
            output_pos: utils::read_u32(&mut rdr)?,
        })
    }
}

fn validate(states: &[State], outputs: &[Output], is_leftmost: bool) -> bool {
    // Checks the following constraints:
    //  - state.check() < states.len()
    //  - state.fail() < states.len()
    //  - state.output_pos() < outputs.len()
    for state in states {
        if state.check() as usize >= states.len() {
            return false;
        }
        if state.fail() as usize >= states.len() && !(is_leftmost && state.fail() == DEAD_STATE_IDX)
        {
            return false;
        }
        if let Some(output_pos) = state.output_pos() {
            if output_pos as usize >= outputs.len() {
                return false;
            }
        }
    }

    // Checks that the length is less than or equal to the depth.
    let mut stack = vec![];
    let mut depths = vec![0; states.len()];
    for (mut state_id, mut state) in states.iter().enumerate() {
        if state.check() == DEAD_STATE_IDX {
            continue;
        }
        stack.clear();
        let depth_base = loop {
            let depth = unsafe { depths.get_unchecked_mut(state_id) };
            if state_id == 0 || *depth != 0 {
                break *depth;
            }
            *depth = 1;
            let child_state_id = state_id;
            state_id = state.check() as usize;
            state = unsafe { states.get_unchecked(state_id) };
            if let Some(base) = state.base() {
                stack.push((child_state_id, (child_state_id as i32 - base) as u32));
            } else {
                return false;
            }
        };
        let mut len_utf8 = 0;
        for &(state_id, c) in stack.iter().rev() {
            unsafe {
                if let Some(c) = char::from_u32(c) {
                    len_utf8 += c.len_utf8();
                } else {
                    return false;
                }
                *depths.get_unchecked_mut(state_id) = depth_base + len_utf8;
            }
        }
    }
    for (state, depth) in states.iter().zip(&depths) {
        if state.check() == DEAD_STATE_IDX {
            continue;
        }
        if let Some(output_pos) = state.output_pos() {
            unsafe {
                if outputs.get_unchecked(output_pos as usize).length() as usize > *depth {
                    return false;
                }
            }
        }
    }

    // Checks that suffix patterns have shorter lengths.
    let mut prev_length = 0;
    for output in outputs {
        let length = output.length();
        if !output.is_begin() && length >= prev_length {
            return false;
        }
        prev_length = length;
    }
    true
}
