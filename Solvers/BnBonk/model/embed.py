class EmbSfc:
    def __init__(self) -> None:
        self.__links = []
        self.__nodes = []
    
    def is_exists_node(self, node) -> bool:
        if node in self.__nodes:
            return True
        return False
    
    def add_node(self, node):
        self.__nodes.append(node)
    
    def remove_node(self, node):
        self.__nodes.remove(node)
    
    def add_link(self, link):
        self.__links.append(link)
    
    def remove_link(self, link):
        self.__links.remove(link)
    
    def remove_links(self, links):
        if len(links) == 0:
            return
        for link in links:
            self.remove_link(link)
    
    def links(self) -> list:
        return self.__links
