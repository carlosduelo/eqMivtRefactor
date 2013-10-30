/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <linkedList.h>

#include <iostream>

namespace eqMivt
{

template<>
bool LinkedList<int>::reSize(int size)
{
	if (memoryList != 0)
		delete[] memoryList;

	freePositions 	= size;
	memoryList 	= new NodeLinkedList<int>[size];
	if (memoryList == 0)
		return false;
	list 		= memoryList;
	last 		= &memoryList[size-1];

	if (list == last)
	{
		memoryList[0].after 	= 0;
		memoryList[0].before 	= 0;
		memoryList[0].element 	= 0;
		memoryList[0].id		= -1;
		memoryList[0].refs		= 0;
		return true;
	}

	for(int i=0; i<size; i++)
	{
		if (i==0)
		{
			memoryList[i].after 	= &memoryList[i+1];
			memoryList[i].before 	= 0;
			memoryList[i].element 	= i;
			memoryList[i].id		= -1;
			memoryList[i].refs		= 0;
		}
		else if (i==size-1)
		{
			memoryList[i].after 	= 0;
			memoryList[i].before 	= &memoryList[i-1];
			memoryList[i].element 	= i;
			memoryList[i].id		= -1;
			memoryList[i].refs		= 0;
		}
		else
		{
			memoryList[i].after 	= &memoryList[i+1];
			memoryList[i].before 	= &memoryList[i-1];
			memoryList[i].element 	= i;
			memoryList[i].id		= -1;
			memoryList[i].refs		= 0;
		}
	}

	return true;
}

template<>
bool LinkedList<index_node_t>::reSize(int size)
{
	if (memoryList != 0)
		delete[] memoryList;

	freePositions 	= size;
	memoryList 	= new NodeLinkedList<index_node_t>[size];
	if (memoryList == 0)
		return false;
	list 		= memoryList;
	last 		= &memoryList[size-1];

	if (list == last)
	{
		memoryList[0].after 	= 0;
		memoryList[0].before 	= 0;
		memoryList[0].element 	= 0;
		memoryList[0].id		= 0;
		memoryList[0].refs		= 0;
		return true;
	}

	for(int i=0; i<size; i++)
	{
		if (i==0)
		{
			memoryList[i].after 	= &memoryList[i+1];
			memoryList[i].before 	= 0;
			memoryList[i].element 	= i;
			memoryList[i].id		= 0;
			memoryList[i].refs		= 0;
		}
		else if (i==size-1)
		{
			memoryList[i].after 	= 0;
			memoryList[i].before 	= &memoryList[i-1];
			memoryList[i].element 	= i;
			memoryList[i].id		= 0;
			memoryList[i].refs		= 0;
		}
		else
		{
			memoryList[i].after 	= &memoryList[i+1];
			memoryList[i].before 	= &memoryList[i-1];
			memoryList[i].element 	= i;
			memoryList[i].id		= 0;
			memoryList[i].refs		= 0;
		}
	}

	return true;
}

template<>
NodeLinkedList<int> * LinkedList<int>::getFirstFreePosition()
{
	NodeLinkedList<int> * first = list;

	if (freePositions == 0)
	{
		if (list == last)
		{
			if (list->refs != 0)
				return 0;
		}
		else
		{
			// Search first free position
			while(list->refs != 0)
			{
				moveToLastPosition(list);
				if (first == list)
				{
					return 0;
				}
			}
		}
	}
	else
	{
		if (list == last)
		{
			if (list->refs != 0 && list->id >= 0)
				throw;
		}
		else
		{
			// Search first free position
			while(list->refs != 0 && list->id >= 0)
			{
				moveToLastPosition(list);
				if (first == list)
				{
					std::cerr<<"Error cache is unistable"<<std::endl;
					throw;
				}
			}

		}
		freePositions--;
	}
	return list;
}

template<>
NodeLinkedList<index_node_t> * LinkedList<index_node_t>::getFirstFreePosition()
{
	NodeLinkedList<index_node_t> * first = list;

	if (freePositions == 0)
	{
		if (list == last)
		{
			if (list->refs != 0)
				return 0;
		}
		else
		{
			// Search first free position
			while(list->refs != 0)
			{
				moveToLastPosition(list);
				if (first == list)
				{
					return 0;
				}
			}
		}
	}
	else
	{
		if (list == last)
		{
			if (list->refs != 0 && list->id > 0)
				throw;
		}
		else
		{
			// Search first free position
			while(list->refs != 0 && list->id > 0)
			{
				moveToLastPosition(list);
				if (first == list)
				{
					std::cerr<<"Error cache is unistable"<<std::endl;
					throw;
				}
			}
		}

		freePositions--;
	}
	return list;
}

template<class T>
NodeLinkedList<T> * LinkedList<T>::moveToLastPosition(NodeLinkedList<T> * node)
{
	if (list == last)
	{
		return list;
	}
	else if (node == list)
	{
		NodeLinkedList<T> * first = list;

		list = first->after;
		list->before = 0;
		
		first->after  = 0;
		first->before = last;
		
		last->after = first;
		
		last = first;

		return first;
	}
	else if (node == last)
	{
		return node;
	}
	else
	{
		node->before->after = node->after;
		node->after->before = node->before;
		
		last->after = node;
		
		node->before = last;
		node->after  = 0;
		last = node;
		
		return node;
	}
}
}
