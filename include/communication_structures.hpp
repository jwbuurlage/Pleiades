namespace pleiades {

struct scanline {
    int begin;
    int count;
}

struct faces {
    std::vector<int> contributors;
    std::vector<scanline> scanlines;
}

} // namespace pleiades
