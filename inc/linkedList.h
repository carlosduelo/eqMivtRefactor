/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_LINKED_LIST_H
#define EQ_MIVT_LINKED_LIST_H

#include <typedef.h>

#include <vector>

namespace eqMivt
{
class NodeLinkedList
{
	public:
		NodeLinkedList * 	after;
		NodeLinkedList * 	before;
		unsigned int	 	element;
		index_node_t 	 	id;
		int					refs;
		std::vector<int>	pendingPlanes;
};

class LinkedList
{
	private:
		NodeLinkedList * 	list;
		NodeLinkedList * 	last;
		NodeLinkedList * 	memoryList;
		int			freePositions;

	public:
		LinkedList();
		
		~LinkedList();

		void reSize(int size);

		/* pop_front and push_last */
		NodeLinkedList * 	getFirstFreePosition();

		NodeLinkedList * 	moveToLastPosition(NodeLinkedList * node);	
};
}
#endif /*EQ_MIVT_LINKED_LIST_H*/
