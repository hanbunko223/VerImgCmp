package bitset

type BitSet struct {
	words []uint64
}

func New(length uint) *BitSet {
	wordCount := 0
	if length > 0 {
		wordCount = int((length + 63) / 64)
	}
	return &BitSet{words: make([]uint64, wordCount)}
}

func (b *BitSet) Set(i uint) *BitSet {
	word := i / 64
	bit := i % 64
	if int(word) >= len(b.words) {
		extended := make([]uint64, word+1)
		copy(extended, b.words)
		b.words = extended
	}
	b.words[word] |= uint64(1) << bit
	return b
}

func (b *BitSet) Test(i uint) bool {
	word := i / 64
	bit := i % 64
	if int(word) >= len(b.words) {
		return false
	}
	return (b.words[word] & (uint64(1) << bit)) != 0
}
