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

template < class T >
class NodeLinkedList
{
	public:
		NodeLinkedList<T> * 	after;
		NodeLinkedList<T> * 	before;
		unsigned int			element;
		T						id;
		int						refs;
};

template <>
class NodeLinkedList<int>
{
	public:
		NodeLinkedList<int> * 	after;
		NodeLinkedList<int> * 	before;
		unsigned int			element;
		int						id;
		int						refs;
};
template <>
class NodeLinkedList<index_node_t>
{
	public:
		NodeLinkedList<index_node_t> * 	after;
		NodeLinkedList<index_node_t> * 	before;
		unsigned int			element;
		index_node_t			id;
		int						refs;
		std::vector<int>		pendingPlanes;
};


typedef NodeLinkedList<index_node_t> NodeCube_t;
typedef NodeLinkedList<int> NodePlane_t;

template < class T >
class LinkedList
{
	private:
		NodeLinkedList<T> * 	list;
		NodeLinkedList<T> * 	last;
		NodeLinkedList<T> * 	memoryList;
		int			freePositions;

	public:
		LinkedList()
		{
			list = 0;
			last = 0;
			memoryList = 0;
			freePositions = 0;
		}
		
		~LinkedList()
		{
			if (memoryList != 0)
				delete[] memoryList;
		}

		void reSize(int size);

		/* pop_front and push_last */
		NodeLinkedList<T> * 	getFirstFreePosition();

		NodeLinkedList<T> * 	moveToLastPosition(NodeLinkedList<T> * node);	
};
}
#endif /*EQ_MIVT_LINKED_LIST_H*/
