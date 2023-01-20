use crate::bitmap::Bitmap64;

/// A Bit-Parallel-XChecker allows for X_CHECK (find_base) operation faster than a simple double-loop (brute-force with empty-linking-method) algorithm.
/// In addition, BPXCheck can be easily combined with Empty-Linking-Method.
/// # Complexity
/// This algorithm, named Bit-Parallel-XCheck, collectively verifies W-adjacent "base" candidates in O( C log W ) times
///   where W is word length of calculator, a.k.a. 64,
///   and C is number of children of an target node.
/// # Space
/// This algorithm require 2N bit additional space where N is length of a Double-Array.
#[derive(Default)]
pub struct BPXChecker {
    map: Bitmap64,
    basemap: Bitmap64,
}

impl BPXChecker {
    pub const fn new_empty() -> Self {
        Self {
            map: Bitmap64::new_empty(),
            basemap: Bitmap64::new_empty(),
        }
    }

    pub fn new(len: usize) -> Self {
        Self {
            map: Bitmap64::new(len),
            basemap: Bitmap64::new(len),
        }
    }

    pub fn state_map_len(&self) -> usize {
        self.map.len()
    }

    pub fn resize_state_map(&mut self, new_len: usize) {
        self.map.resize(new_len);
    }

    pub fn resize_base_map(&mut self, new_len: usize) {
        self.basemap.resize(new_len);
    }

    #[inline(always)]
    pub fn get_base_mask(&self, wi: usize) -> u64 {
        self.basemap.get_word(wi)
    }

    #[inline(always)]
    pub fn get_state_mask(&self, wi: usize) -> u64 {
        self.map.get_word(wi)
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn get_state_mask_from(&self, i: usize) -> u64 {
        self.map.get_word_from(i)
    }

    #[inline(always)]
    pub fn set_base_word(&mut self, word_idx: usize, word: u64) {
        self.basemap.set_word_direct(word_idx, word);
    }

    #[inline(always)]
    pub fn set_state_word(&mut self, word_idx: usize, word: u64) {
        self.map.set_word_direct(word_idx, word);
    }

    #[inline(always)]
    pub fn base_fixed(&self, i: usize) -> bool {
        self.basemap.get_bit(i)
    }

    #[inline(always)]
    pub fn state_fixed(&self, i: usize) -> bool {
        self.map.get_bit(i)
    }

    #[inline(always)]
    pub fn set_base_fixed(&mut self, i: usize) {
        self.basemap.set_bit(i, true);
    }

    #[inline(always)]
    pub fn set_state_fixed(&mut self, i: usize) {
        self.map.set_bit(i, true);
    }

    #[inline(always)]
    pub fn reset_base(&mut self, i: usize) {
        self.basemap.set_bit(i, false);
    }

    #[inline(always)]
    pub fn reset_state(&mut self, i: usize) {
        self.map.set_bit(i, false);
    }

    #[inline(always)]
    pub fn set_64states_fixed(&mut self, wi: usize) {
        self.map.set_word(wi, true);
    }

    #[inline(always)]
    pub fn reset_64elements(&mut self, wi: usize) {
        self.map.set_word(wi, false);
        self.basemap.set_word(wi, false);
    }

    #[inline(always)]
    pub fn disabled_base_mask_xor<'a, I, T>(&self, base_front: u64, label_iter: I) -> u64
        where I: Iterator<Item = &'a T>,
              T: Copy + 'a,
              u32: From<T>,
              u64: From<T>
    {
        self._disabled_base_mask_xor(base_front, label_iter, 0)
    }

    #[inline(always)]
    pub fn disabled_base_mask_xor_unique_base<'a, I, T>(&self, base_front: u64, label_iter: I) -> u64
        where I: Iterator<Item = &'a T>,
              T: Copy + 'a,
              u32: From<T>,
              u64: From<T>
    {
        self._disabled_base_mask_xor(base_front, label_iter,
                                     self.basemap.get_word(Bitmap64::word_index(base_front as usize)))
    }

    const BLOCK_MASKS: [u64; 6] = [
        0b0101u64 * 0x1111111111111111u64,
        0b0011u64 * 0x1111111111111111u64,
        0x0F0F0F0F0F0F0F0Fu64,
        0x00FF00FF00FF00FFu64,
        0x0000FFFF0000FFFFu64,
        0x00000000FFFFFFFFu64, // never used
    ];
    pub const NO_CANDIDATE: u64 = Bitmap64::word_filled_by(true);

    #[inline(always)]
    fn _disabled_base_mask_xor<'a, I, T>(&self, base_front: u64, label_iter: I, init_mask: u64) -> u64
        where
            I: Iterator<Item = &'a T>,
            T: Copy + 'a,
            u32: From<T>,
            u64: From<T>
    {
        let mut x = init_mask;
        for &label in label_iter {
            let w = {
                let q = Bitmap64::word_index((base_front ^ u64::from(label)) as usize);
                let mut w: u64 = self.map.get_word(q);
                // Block-wise swap
                for i in 0..5 {
                    let width = 1u32 << i;
                    if u32::from(label) & width != 0 {
                        w = ((w >> width) & BPXChecker::BLOCK_MASKS[i]) | ((w & BPXChecker::BLOCK_MASKS[i]) << width);
                    }
                }
                if u32::from(label) & (1u32 << 5) != 0 {
                    w = (w >> 32) | (w << 32);
                }
                w
            };
            // Merge invalid base mask
            x |= w;
            if x == BPXChecker::NO_CANDIDATE {
                return BPXChecker::NO_CANDIDATE;
            }
        }
        x
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn disabled_base_mask_plus<'a, I, T>(&self, base_front: i32, label_iter: I) -> u64
        where
            I: Iterator<Item = &'a T>,
            T: Copy + Into<i32> + 'a
    {
        self._disabled_base_mask_plus(base_front, label_iter, 0u64)
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn disabled_base_mask_plus_unique_base<'a, I, T>(&self, base_front: i32, label_iter: I) -> u64
    where
        I: Iterator<Item = &'a T>,
        T: Copy + Into<i32> + 'a
    {
        self._disabled_base_mask_plus(base_front, label_iter, self.basemap.get_word_from(base_front as usize))
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn _disabled_base_mask_plus<'a, I, T>(&self, base_front: i32, label_iter: I, init_mask: u64) -> u64
        where
            I: Iterator<Item = &'a T>,
            T: Copy + Into<i32> + 'a
    {
        let mut x = init_mask;
        for &label in label_iter {
            let w = self.map.get_word_from((base_front + label.into()) as usize);
            x |= w;
            if x == BPXChecker::NO_CANDIDATE {
                return BPXChecker::NO_CANDIDATE;
            }
        }
        x
    }

    const BASE_FRONT_MASK: u64 = !(Bitmap64::BITS - 1) as u64;

    /// Find valid base for length of iteration `label_iter`
    /// in 64 adjacent 64-aligned bases including `base_origin` collectively.
    /// Return:
    ///     minimum value of valid bases
    /// Complexity:
    ///     O(|labels| log w) where w is word size (w=64)
    #[inline(always)]
    pub fn find_base_for_64adjacent_xor(&self, base_origin: u64, edges: &[(u32, u32)]) -> Option<u64> {
        let base_front = base_origin & BPXChecker::BASE_FRONT_MASK;
        let x = self.disabled_base_mask_xor(base_front,
                                            FirstIterator::new(edges));
        if x != BPXChecker::NO_CANDIDATE {
            // Return minimum base of valid bases
            Some(base_front ^ (x.trailing_ones() as u64))
        } else {
            None
        }
    }
    /// Find valid base for length of iteration `label_iter`
    /// in 64 adjacent 64-aligned bases including `base_origin` collectively.
    /// Returned base is unique.
    /// Return:
    ///     minimum value of valid and unique bases
    /// Complexity: 
    ///     O(|labels| log w) where w is word size (w=64)
    #[inline(always)]
    pub fn find_unique_base_for_64adjacent_xor(&self, base_origin: u64, labels: &[u8]) -> Option<u64> {
        let base_front = base_origin & BPXChecker::BASE_FRONT_MASK;
        let x = self.disabled_base_mask_xor_unique_base(base_front, labels.into_iter());
        if x != BPXChecker::NO_CANDIDATE {
            // Return minimum base of valid bases
            Some(base_front ^ (x.trailing_ones() as u64))
        } else {
            None
        }
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn find_base_for_64adjacent_plus(&self, base_origin: i32, edges: &[(i32, u32)]) -> Option<i32> {
        let x = self.disabled_base_mask_plus(base_origin,
                                             FirstIterator::new(edges));
        if x != BPXChecker::NO_CANDIDATE {
            Some(base_origin + x.trailing_ones() as i32) // Return one of the candidates
        } else {
            None
        }
    }
}

struct FirstIterator<'a, T> {
    arr: &'a [(T, u32)],
    idx: usize,
}

impl<T> FirstIterator<'_, T> {
    pub fn new(edges: &[(T, u32)]) -> FirstIterator<'_, T> {
        FirstIterator {
            arr: edges,
            idx: 0usize,
        }
    }
}

impl<'a, T> Iterator for FirstIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.arr.len() {
            return None;
        }
        let cur = &self.arr[self.idx].0;
        self.idx += 1;
        Some(cur)
    }
}

#[cfg(test)]
use alloc::vec::Vec;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_base_for_64adjacent_xor() {
        let map: [usize; 64] = [1,0,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1];
        let labels: Vec<u8> = vec![1, 3, 7, 9, 11, 23, 41];
        let expected_bases: [usize; 6] = [6, 14, 37, 45, 51, 57];

        let mut xc = BPXChecker::new(64);
        for i in 0..64 {
            if map[i] != 0 {
                xc.set_state_fixed(i);
            }
        }
        let x = xc.disabled_base_mask_xor(0, (&labels).into_iter());
        let mut base_candidates = vec![];
        for i in 0..64 {
            if x & (1u64 << i) == 0 {
                base_candidates.push(i);
            }
        }
        assert_eq!(expected_bases.len(), base_candidates.len());
        for i in 0..expected_bases.len() {
            assert_eq!(expected_bases[i], base_candidates[i]);
        }
    }

    #[test]
    fn test_find_base_with_masked_base_xor() {
        let map: [usize; 64] = [1,0,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1];
        let labels: Vec<u8> = vec! [1, 3, 7, 9, 11, 23, 41];
        let invalid_bases = {
            let mut arr: Vec<usize> = vec![];
            let mut i = 0;
            loop {
                arr.push(i);
                i += 2;
                if i >= 64 { break; }
            }
            arr
        };
        let expected_bases: [usize; 4] = [37, 45, 51, 57];

        let mut xc = BPXChecker::new(64);
        for i in 0..64 {
            if map[i] != 0 {
                xc.set_state_fixed(i);
            }
        }
        for b in invalid_bases {
            xc.set_base_fixed(b);
        }
        let x = xc.disabled_base_mask_xor_unique_base(0, (&labels).into_iter());
        let mut base_candidates = vec![];
        for i in 0..64 {
            if x & (1u64 << i) == 0 {
                base_candidates.push(i);
            }
        }
        assert_eq!(expected_bases.len(), base_candidates.len());
        for i in 0..expected_bases.len() {
            assert_eq!(expected_bases[i], base_candidates[i]);
        }
    }

    #[test]
    fn test_find_base_for_64adjacent_plus() {
        let map: [usize; 64] = [1,0,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1];
        let labels: Vec<u8> = vec![1, 3, 7, 9, 11, 23, 41];
        let expected_bases: [u64; 15] = [6, 25, 31, 33, 35, 39, 41, 43, 45, 46, 47, 55, 58, 61, 63];

        let mut xc = BPXChecker::new(64);
        for i in 0..64 {
            if map[i] != 0 {
                xc.set_state_fixed(i);
            }
        }
        let x = xc.disabled_base_mask_plus_unique_base(0, (&labels).into_iter());
        let mut base_candidates = vec![];
        for i in 0..64 {
            if x & (1u64 << i) == 0 {
                base_candidates.push(i);
            }
        }
        assert_eq!(expected_bases.len(), base_candidates.len());
        for i in 0..expected_bases.len() {
            assert_eq!(expected_bases[i], base_candidates[i]);
        }
    }
}
